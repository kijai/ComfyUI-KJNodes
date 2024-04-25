import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "KJNodes.jsnodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData.name) {
			case "ConditioningMultiCombine":
				nodeType.prototype.onNodeCreated = function () {
				this.cond_type = "CONDITIONING"
				this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
					    if(target_number_of_inputs===this.inputs.length)return; // already set, do nothing

					    if(target_number_of_inputs < this.inputs.length){
    						for(let i = this.inputs.length; i>=this.inputs_offset+target_number_of_inputs; i--)
							      this.removeInput(i)
					    }
                        else{
						    for(let i = this.inputs.length+1-this.inputs_offset; i <= target_number_of_inputs; ++i)
						    	this.addInput(`conditioning_${i}`, this.cond_type)
                        }
				});
				}
				break;
				case "ImageBatchMulti":
					nodeType.prototype.onNodeCreated = function () {
					this._type = "IMAGE"
					this.inputs_offset = nodeData.name.includes("selective")?1:0
					this.addWidget("button", "Update inputs", null, () => {
						if (!this.inputs) {
							this.inputs = [];
						}
						const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
							if(target_number_of_inputs===this.inputs.length)return; // already set, do nothing
	
							if(target_number_of_inputs < this.inputs.length){
								for(let i = this.inputs.length; i>=this.inputs_offset+target_number_of_inputs; i--)
									  this.removeInput(i)
							}
							else{
								for(let i = this.inputs.length+1-this.inputs_offset; i <= target_number_of_inputs; ++i)
									this.addInput(`image_${i}`, this._type)
							}
					});
					}
					break;
			case "SoundReactive":
				nodeType.prototype.onNodeCreated = function () {
					let audioContext;
					let microphoneStream;
					let animationFrameId;
					let analyser;
					let dataArray;
					let startRangeHz;
    				let endRangeHz;
					let smoothingFactor = 0.5;
					let smoothedSoundLevel = 0;
				
					// Function to update the widget value in real-time
					const updateWidgetValueInRealTime = () => {
						// Ensure analyser and dataArray are defined before using them
						if (analyser && dataArray) {
							analyser.getByteFrequencyData(dataArray);

							const startRangeHzWidget = this.widgets.find(w => w.name === "start_range_hz");
							if (startRangeHzWidget) startRangeHz = startRangeHzWidget.value;
							const endRangeHzWidget = this.widgets.find(w => w.name === "end_range_hz");
							if (endRangeHzWidget) endRangeHz = endRangeHzWidget.value;
							const smoothingFactorWidget = this.widgets.find(w => w.name === "smoothing_factor");
							if (smoothingFactorWidget) smoothingFactor = smoothingFactorWidget.value;

							// Calculate frequency bin width (frequency resolution)
							const frequencyBinWidth = audioContext.sampleRate / analyser.fftSize;	
							// Convert the widget values from Hz to indices
							const startRangeIndex = Math.floor(startRangeHz / frequencyBinWidth);
							const endRangeIndex = Math.floor(endRangeHz / frequencyBinWidth);

							// Function to calculate the average value for a frequency range
							const calculateAverage = (start, end) => {
								const sum = dataArray.slice(start, end).reduce((acc, val) => acc + val, 0);
								const average = sum / (end - start);

								// Apply exponential moving average smoothing
    							smoothedSoundLevel = (average * (1 - smoothingFactor)) + (smoothedSoundLevel * smoothingFactor);
								return smoothedSoundLevel;
							};
							// Calculate the average levels for each frequency range
							const soundLevel = calculateAverage(startRangeIndex, endRangeIndex);
							
							// Update the widget values

							const lowLevelWidget = this.widgets.find(w => w.name === "sound_level");
							if (lowLevelWidget) lowLevelWidget.value = soundLevel;

							animationFrameId = requestAnimationFrame(updateWidgetValueInRealTime);
						}
					};
				
					// Function to start capturing audio from the microphone
					const startMicrophoneCapture = () => {
						// Only create the audio context and analyser once
						if (!audioContext) {
							audioContext = new (window.AudioContext || window.webkitAudioContext)();
							// Access the sample rate of the audio context
							console.log(`Sample rate: ${audioContext.sampleRate}Hz`);
							analyser = audioContext.createAnalyser();
							analyser.fftSize = 2048;
							dataArray = new Uint8Array(analyser.frequencyBinCount);
							// Get the range values from widgets (assumed to be in Hz)
							const lowRangeWidget = this.widgets.find(w => w.name === "low_range_hz");
							if (lowRangeWidget) startRangeHz = lowRangeWidget.value;
				
							const midRangeWidget = this.widgets.find(w => w.name === "mid_range_hz");
							if (midRangeWidget) endRangeHz = midRangeWidget.value;
						}
						
						navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
							microphoneStream = stream;
							const microphone = audioContext.createMediaStreamSource(stream);
							microphone.connect(analyser);
							updateWidgetValueInRealTime();
						}).catch(error => {
							console.error('Access to microphone was denied or an error occurred:', error);
						});
					};
				
					// Function to stop capturing audio from the microphone
					const stopMicrophoneCapture = () => {
						if (animationFrameId) {
							cancelAnimationFrame(animationFrameId);
						}
						if (microphoneStream) {
							microphoneStream.getTracks().forEach(track => track.stop());
						}
						if (audioContext) {
							audioContext.close();
							// Reset audioContext to ensure it can be created again when starting
							audioContext = null;
						}
					};
				
					// Add start button
					this.addWidget("button", "Start mic capture", null, startMicrophoneCapture);
				
					// Add stop button
					this.addWidget("button", "Stop mic capture", null, stopMicrophoneCapture);
				};
			break;
			
	}	
		// to keep Set/Get node virtual connections visible when offscreen
		const originalComputeVisibleNodes = LGraphCanvas.prototype.computeVisibleNodes;
		LGraphCanvas.prototype.computeVisibleNodes = function (nodes, out) {
			const visibleNodes = originalComputeVisibleNodes.apply(this, arguments);
			const setAndGetNodes = this.graph._nodes.filter(node => node.type === "SetNode" || node.type === "GetNode");
			for (const node of setAndGetNodes) {
				if (!visibleNodes.includes(node) && node.drawConnection) {
					visibleNodes.push(node);
				}
			}
			return visibleNodes;
		};
	},
});