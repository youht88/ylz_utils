from langchain_community.document_loaders.image import UnstructuredImageLoader

class ImageLoader:
     @classmethod
     def loader(cls,filename):
         "./example_data/layout-parser-paper-screenshot.png"
         return UnstructuredImageLoader(filename)