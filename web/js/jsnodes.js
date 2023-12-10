import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "KJNodes.ConditioningMultiCombine",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData.name) {
			case "ConditioningMultiCombine":
				nodeType.prototype.onNodeMoved = function () {
					console.log(this.pos[0])
				}
				nodeType.prototype.onNodeCreated = function () {
				//this.inputs_offset = nodeData.name.includes("selective")?1:0
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
			case "SoundReactive":
				nodeType.prototype.onNodeCreated = function () {
					let audioContext;
					let microphoneStream;
					let animationFrameId;
					let analyser;
					let dataArray;
					let lowRangeHz;
    				let midRangeHz;
				
					// Function to update the widget value in real-time
					const updateWidgetValueInRealTime = () => {
						// Ensure analyser and dataArray are defined before using them
						if (analyser && dataArray) {
							analyser.getByteFrequencyData(dataArray);

							// Calculate frequency bin width (frequency resolution)
							const frequencyBinWidth = audioContext.sampleRate / analyser.fftSize;	
							// Convert the widget values from Hz to indices
							const lowRangeIndex = Math.floor(lowRangeHz / frequencyBinWidth);
							const midRangeIndex = Math.floor(midRangeHz / frequencyBinWidth);

							// Define frequency ranges for low, mid, and high
							const frequencyRanges = {
								low: { start: 0, end: lowRangeIndex },
								mid: { start: lowRangeIndex, end: midRangeIndex },
								high: { start: midRangeIndex, end: dataArray.length }
							};	
							const lowRangeHzWidget = this.widgets.find(w => w.name === "low_range_hz");
							if (lowRangeHzWidget) lowRangeHz = lowRangeHzWidget.value;

							const midRangeHzWidget = this.widgets.find(w => w.name === "mid_range_hz");
							if (midRangeHzWidget) midRangeHz = midRangeHzWidget.value;

							// Function to calculate the average value for a frequency range
							const calculateAverage = (start, end) => {
								const sum = dataArray.slice(start, end).reduce((acc, val) => acc + val, 0);
								return sum / (end - start);
							};
							// Calculate the average levels for each frequency range
							const lowLevel = calculateAverage(frequencyRanges.low.start, frequencyRanges.low.end);
							const midLevel = calculateAverage(frequencyRanges.low.end, frequencyRanges.mid.end); // mid starts where low ends
							const highLevel = calculateAverage(frequencyRanges.mid.end, frequencyRanges.high.end); // high starts where mid ends
							const averageLevel = dataArray.reduce((sum, averageLevel) => sum + averageLevel, 0) / dataArray.length;
							
							// Update the widget values
							const averageLevelWidget = this.widgets.find(w => w.name === "average_level");
							if (averageLevelWidget) averageLevelWidget.value = averageLevel;

							const lowLevelWidget = this.widgets.find(w => w.name === "low_level");
							if (lowLevelWidget) lowLevelWidget.value = lowLevel;

							const midLevelWidget = this.widgets.find(w => w.name === "mid_level");
							if (midLevelWidget) midLevelWidget.value = midLevel;

							const highLevelWidget = this.widgets.find(w => w.name === "high_level");
							if (highLevelWidget) highLevelWidget.value = highLevel;

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
							if (lowRangeWidget) lowRangeHz = lowRangeWidget.value;
				
							const midRangeWidget = this.widgets.find(w => w.name === "mid_range_hz");
							if (midRangeWidget) midRangeHz = midRangeWidget.value;
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
	},
});