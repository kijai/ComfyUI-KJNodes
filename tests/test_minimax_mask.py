import importlib.util
import asyncio
from pathlib import Path
import unittest

import torch
from comfy.cli_args import args

args.cpu = True

from comfy.nested_tensor import NestedTensor
from comfy.sampler_helpers import prepare_mask
from comfy.utils import pack_latents, unpack_latents
import nodes
from server import PromptServer


spec = importlib.util.spec_from_file_location("kj_minimax_nodes", Path(__file__).parents[1] / "nodes/minimax_nodes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MiniMaxH3AudioVideoMask = module.MiniMaxH3AudioVideoMask


class MiniMaxMaskTests(unittest.TestCase):
    def latent(self):
        return {"samples": NestedTensor((torch.randn(1, 24, 37, 2, 4), torch.randn(1, 32, 2, 207))), "test_metadata": "keep"}

    def run_node(self, latent, video=(0, 0), audio=(2, 4), mode="partial", combine="overwrite"):
        return MiniMaxH3AudioVideoMask.execute(latent, *video, *audio, mode, combine)[0]

    def test_audio_only_preserves_video_and_input(self):
        latent = self.latent()
        output = self.run_node(latent)
        video, audio = output["noise_mask"].unbind()
        self.assertEqual(output["test_metadata"], "keep")
        self.assertNotIn("noise_mask", latent)
        self.assertIs(output["samples"].tensors[0], latent["samples"].tensors[0])
        self.assertIs(output["samples"].tensors[1], latent["samples"].tensors[1])
        self.assertEqual(torch.count_nonzero(video).item(), 0)
        expected = torch.zeros(1, 1, 2, 207)
        expected[..., 80:160] = 1
        torch.testing.assert_close(audio, expected)

    def test_inward_video_boundaries_and_empty_audio(self):
        output = self.run_node(self.latent(), video=(1, 2), audio=(0, 0))
        video, audio = output["noise_mask"].unbind()
        expected = torch.zeros(1, 1, 37, 2, 4)
        # Whole tokens in [24, 48) pixel frames start at 26, 30, 34, 35, 39, 43, 47.
        # The token starting at 47 ends at 51 and must remain untouched.
        expected[:, :, 8:14] = 1
        torch.testing.assert_close(video, expected)
        self.assertEqual(torch.count_nonzero(audio).item(), 0)

    def test_both_streams_and_exact_audio_boundary(self):
        output = self.run_node(self.latent(), video=(1 / 24, 5 / 24), audio=(4.025, 4.075))
        video, audio = output["noise_mask"].unbind()
        self.assertEqual(video.sum().item(), 8)
        self.assertTrue(torch.all(video[:, :, 1] == 1))
        expected = torch.zeros(1, 1, 2, 207)
        expected[..., 161:163] = 1
        torch.testing.assert_close(audio, expected)

    def test_existing_spatial_and_soft_masks(self):
        for combine in ("add", "subtract", "overwrite"):
            with self.subTest(combine=combine):
                latent = self.latent()
                vm = torch.full((1, 1, 37, 2, 4), 0.25)
                vm[..., 0] = 0.75
                am = torch.full((1, 1, 2, 207), 0.5)
                latent["noise_mask"] = NestedTensor((vm, am))
                original_v, original_a = vm.clone(), am.clone()
                output = self.run_node(latent, video=(1 / 24, 5 / 24), combine=combine)
                expected_v = torch.zeros_like(vm) if combine == "overwrite" else vm.clone()
                expected_a = torch.zeros_like(am) if combine == "overwrite" else am.clone()
                expected_v[:, :, 1:2] = 0 if combine == "subtract" else 1
                expected_a[..., 80:160] = 0 if combine == "subtract" else 1
                torch.testing.assert_close(output["noise_mask"].tensors[0], expected_v)
                torch.testing.assert_close(output["noise_mask"].tensors[1], expected_a)
                torch.testing.assert_close(vm, original_v)
                torch.testing.assert_close(am, original_a)

    def test_subtract_without_mask_starts_from_zero(self):
        output = self.run_node(self.latent(), video=(0, 5), combine="subtract")
        for mask in output["noise_mask"].unbind():
            self.assertEqual(torch.count_nonzero(mask).item(), 0)

    def test_pad_uses_common_duration_and_always_generates_tail(self):
        for combine in ("add", "subtract", "overwrite"):
            for has_mask in (False, True):
                with self.subTest(combine=combine, has_mask=has_mask):
                    latent = self.latent()
                    if has_mask:
                        # A broadcast mask must be resolved before extending, not stretched across the new duration.
                        latent["noise_mask"] = NestedTensor((torch.full((1, 1, 1, 1, 1), 0.25), torch.full((1, 1, 2, 1), 0.5)))
                    output = self.run_node(latent, video=(0, 0), audio=(10, 10), mode="pad", combine=combine)
                    video, audio = output["samples"].unbind()
                    self.assertEqual(video.shape, (1, 24, 72, 2, 4))
                    self.assertEqual(audio.shape, (1, 32, 2, 405))
                    torch.testing.assert_close(video[:, :, :37], latent["samples"].tensors[0])
                    torch.testing.assert_close(audio[..., :207], latent["samples"].tensors[1])
                    self.assertEqual(torch.count_nonzero(video[:, :, 37:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(audio[..., 207:]).item(), 0)
                    vm, am = output["noise_mask"].unbind()
                    self.assertTrue(torch.all(vm[:, :, 37:] == 1))
                    self.assertTrue(torch.all(am[..., 207:] == 1))
                    keep = has_mask and combine != "overwrite"
                    self.assertTrue(torch.all(vm[:, :, :37] == (0.25 if keep else 0)))
                    self.assertTrue(torch.all(am[..., :207] == (0.5 if keep else 0)))

    def test_truncate_rounds_down_both_streams(self):
        latent = self.latent()
        output = self.run_node(latent, video=(0, 5), audio=(0, 2), mode="truncate")
        video, audio = output["samples"].unbind()
        # 107 frames, 32 video tokens, 178 audio tokens; the next grid length is 124 frames > 5s.
        self.assertEqual(video.shape[2], 32)
        self.assertEqual(audio.shape[-1], 178)
        torch.testing.assert_close(video, latent["samples"].tensors[0][:, :, :32])
        torch.testing.assert_close(audio, latent["samples"].tensors[1][..., :178])
        self.assertNotEqual(video.untyped_storage().data_ptr(), latent["samples"].tensors[0].untyped_storage().data_ptr())
        self.assertTrue(torch.all(output["noise_mask"].tensors[0] == 1))

    def test_length_modes_do_not_resize_in_the_opposite_direction(self):
        latent = self.latent()
        padded = self.run_node(latent, video=(0, 1), audio=(0, 0), mode="pad")
        truncated = self.run_node(latent, video=(0, 10), audio=(0, 0), mode="truncate")
        for output in (padded, truncated):
            for actual, original in zip(output["samples"].unbind(), latent["samples"].unbind()):
                torch.testing.assert_close(actual, original)

    def test_truncate_below_minimum_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "5 frames"):
            self.run_node(self.latent(), video=(0, 0), audio=(0, 0), mode="truncate")

    def test_masks_pass_core_sampler_preparation(self):
        for mode in ("partial", "pad", "truncate"):
            output = self.run_node(self.latent(), video=(1, 10), mode=mode)
            samples = output["samples"].unbind()
            masks = [prepare_mask(mask, sample.shape, "cpu") for mask, sample in zip(output["noise_mask"].unbind(), samples)]
            packed, shapes = pack_latents(samples)
            packed_mask, mask_shapes = pack_latents(masks)
            self.assertEqual(packed_mask.shape, packed.shape)
            self.assertEqual(shapes, mask_shapes)
            for actual, expected in zip(unpack_latents(packed_mask, shapes), masks):
                torch.testing.assert_close(actual, expected)

    def test_empty_short_reversed_and_outside_ranges(self):
        for interval in ((0, 0), (1, 1), (0.05, 0.06), (4, 2), (20, 30)):
            with self.subTest(interval=interval):
                output = self.run_node(self.latent(), video=interval, audio=interval)
                for mask in output["noise_mask"].unbind():
                    self.assertEqual(torch.count_nonzero(mask).item(), 0)

    def test_resize_does_not_move_existing_temporal_mask(self):
        latent = self.latent()
        vm = torch.zeros(1, 1, 37, 2, 4)
        am = torch.zeros(1, 1, 2, 207)
        vm[:, :, 7:12] = 0.375
        am[..., 80:160] = 0.625
        latent["noise_mask"] = NestedTensor((vm, am))
        for mode, interval, vt, at in (("pad", (10, 10), 37, 207), ("truncate", (5, 5), 32, 178)):
            output = self.run_node(latent, video=interval, audio=interval, mode=mode, combine="add")
            torch.testing.assert_close(output["noise_mask"].tensors[0][:, :, :vt], vm[:, :, :vt])
            torch.testing.assert_close(output["noise_mask"].tensors[1][..., :at], am[..., :at])

    def test_exact_valid_length_is_not_rounded_away(self):
        for mode in ("pad", "truncate"):
            output = self.run_node(self.latent(), video=(0, 124 / 24), mode=mode)
            self.assertEqual(output["samples"].tensors[0].shape[2], 37)
            self.assertEqual(output["samples"].tensors[1].shape[-1], 207)

    def test_mismatched_input_durations_resize_together(self):
        latent = self.latent()
        latent["samples"].tensors[1] = torch.zeros(1, 32, 2, 100)
        shortened = self.run_node(latent, video=(0, 5), mode="truncate")
        self.assertEqual(shortened["samples"].tensors[0].shape[2], 17)
        self.assertEqual(shortened["samples"].tensors[1].shape[-1], 93)
        latent["samples"].tensors[1] = torch.zeros(1, 32, 2, 300)
        extended = self.run_node(latent, video=(0, 5), mode="pad")
        self.assertEqual(extended["samples"].tensors[0].shape[2], 57)
        self.assertEqual(extended["samples"].tensors[1].shape[-1], 320)

    def test_resize_preserves_each_stream_dtype_and_device(self):
        for mode in ("pad", "truncate"):
            latent = self.latent()
            latent["samples"].tensors[0] = latent["samples"].tensors[0].to(torch.bfloat16)
            latent["samples"].tensors[1] = latent["samples"].tensors[1].to(torch.float16)
            output = self.run_node(latent, video=(0, 10 if mode == "pad" else 5), mode=mode)
            for actual, original in zip(output["samples"].unbind(), latent["samples"].unbind()):
                self.assertEqual(actual.dtype, original.dtype)
                self.assertEqual(actual.device, original.device)

    def test_wrong_latent_format_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "MiniMax H3 AV latent"):
            self.run_node({"samples": torch.zeros(1, 4, 8, 8)})

    def test_wrong_mask_format_has_clear_error(self):
        latent = self.latent()
        latent["noise_mask"] = torch.zeros(1, 8, 8)
        with self.assertRaisesRegex(ValueError, "NestedTensor"):
            self.run_node(latent, combine="add")

    def test_registration_through_core_loader(self):
        async def load():
            front_end_root = args.front_end_root
            args.front_end_root = str(Path(__file__).parent)
            try:
                PromptServer(asyncio.get_running_loop())
                return await nodes.load_custom_node(str(Path(__file__).resolve().parents[1]))
            finally:
                args.front_end_root = front_end_root

        self.assertTrue(asyncio.run(load()))
        self.assertTrue("LTXVAudioVideoMask" in nodes.NODE_CLASS_MAPPINGS)
        node = nodes.NODE_CLASS_MAPPINGS["MiniMaxH3AudioVideoMask"]
        schema = node.define_schema()
        self.assertEqual(schema.node_id, "MiniMaxH3AudioVideoMask")
        self.assertEqual([item.id for item in schema.inputs], ["av_latent", "video_start_time", "video_end_time", "audio_start_time", "audio_end_time", "max_length", "existing_mask_mode"])
        self.assertEqual(len(schema.outputs), 1)


if __name__ == "__main__":
    unittest.main()
