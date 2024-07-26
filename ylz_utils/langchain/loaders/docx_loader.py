from langchain_community.document_loaders import Docx2txtLoader

class DocxLoader():
    def loader(self, filename: str):
        #"./example_data/fake.docx"
       return Docx2txtLoader(filename)
