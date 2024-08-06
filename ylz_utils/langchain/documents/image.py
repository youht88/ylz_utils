from langchain_community.document_loaders.image import UnstructuredImageLoader

class ImageLib():
     def __init__(self,langchainLib):
          self.langchainLib = langchainLib
     def loader(self,filename):
         "./example_data/layout-parser-paper-screenshot.png"
         return UnstructuredImageLoader(filename)