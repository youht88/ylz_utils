from langchain_community.document_loaders import RecursiveUrlLoader

class UrlLoader():
    def loader(self, url, max_depth=2, extractor=None, metadata_extractor=None):
        #"./example_data/fake.docx"
        loader = RecursiveUrlLoader(
            url = url,
            max_depth= max_depth,
            # use_async=False,
            extractor= extractor,
            metadata_extractor= metadata_extractor
            # exclude_dirs=(),
            # timeout=10,
            # check_response_status=True,
            # continue_on_failure=True,
            # prevent_outside=True,
            # base_url=None,
        )
        return loader.load()