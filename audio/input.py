import sounddevice as sd
from utils.logger import logger
from utils.config import config

class AudioInput:

	def __init__(self,
				 buffer,
				 sample=48000,
				 channels=1,
				 blockSize=1024
				 ):
		
		self.buffer = buffer
		device = getattr(config.device, "input", None)
		if not device:
			try:
				devs = [d['name'] for d in sd.query_devices()]
				if "pulse" in devs:
					device = "pulse"
				elif "default" in devs:
					device = "default"
			except Exception:
				device = None

		self.stream = sd.InputStream(
			device=device,
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