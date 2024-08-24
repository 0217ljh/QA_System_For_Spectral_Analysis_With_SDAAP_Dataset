from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import UnstructuredFileLoader
from Split_and_Load import File_filter

class StrLoader(BaseLoader):

    def __init__(
        self,
        content,
        file_path: None,
        encoding:  None,
        autodetect_encoding: bool = False,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.content=content
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load_and_append(self):
        """Append file."""
        number = len(self.content)
        result = []
        for i in range(number):
            result.append(self.Load(self.content[i], i)[0])
        return result
    def Load(self, text, index):
        """Load from file path."""
        metadata = {"source": self.file_path + '_chunk{}'.format(index)}
        return [Document(page_content=text, metadata=metadata)]

    def load(self):
        """Load from file path."""
        text=self.content
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

def load_file(filepath):
    #暂时只有pdf的一种加载方法，check！
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"): # check
        #loader = UnstructuredFileLoader(filepath)
        #textsplitter = ChineseTextSplitter(pdf=True)
        docs = LangChain_splt.Langchain_pdf(filepath)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs