from os import listdir

base_path = 'circumference_app/text_data'

class Text:
	def __init__(self):

		for filename in listdir(base_path):
			attr_name = filename[:-3]
			with open(f'{base_path}/{filename}') as filestream:
				setattr(self, attr_name, filestream.read())

text = Text()
