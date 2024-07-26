from langchain_community.document_loaders import Docx2txtLoader

class DocxLoader():
    @classmethod
    def loader(cls, filename: str):
        #"./example_data/fake.docx"
       return Docx2txtLoader(filename)
