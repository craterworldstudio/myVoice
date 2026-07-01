import sounddevice as sd
from utils.logger import logger

class AudioInput:

	def __init__(self,
				 buffer,
				 sample=48000,
				 channels=1,
				 blockSize=1024
				 ):
		
		self.buffer = buffer
		self.stream = sd.InputStream(
			samplerate=sample,
			blocksize=blockSize,
			channels=channels,
			callback=self.callback
			)

	def callback(self, indata, frames, time_info, status):
		"""
    	indata:    NumPy array containing the recorded audio chunk. \n
    	frames:    Integer showing the exact number of frames in indata (equals blocksize). \n
    	time_info: CFFI structure containing timestamp data. \n
    	status:    CallbackFlags object indicating errors like overflows.
    	"""

		if status: logger.info(status)

		self.buffer.write(indata)


	def start(self):
		self.stream.start()
	def stop(self):
		self.stream.stop()