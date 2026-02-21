# Man 클래스 정의한 man.py 파일 저장
# 터미널에서 man.py 파일 실행
# python man.py

class Man:
	def __init__(self, name):
		self.name = name
		print("Initialized!") # Initialized!
	
	def hello(self):
		print("Hello " + self.name + "!") # Hello Hwawon!
	
	def goodbye(self):
		print("Good bye " + self.name + "!") # Good bye Hwawon!
	
m = Man("Hwawon")
m.hello()
m.goodbye()