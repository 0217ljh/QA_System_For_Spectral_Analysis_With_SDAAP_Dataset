import tiktoken
class LazyloadTiktoken(object):
    def __init__(self, model):
        self.model = model
        self.tokenizer=tiktoken.encoding_for_model(self.model)
        print('正在加载tokenizer，如果是第一次运行，可能需要一点时间下载参数')
        print('加载tokenizer完毕')
    def get_encoder(self):
        #print('正在加载tokenizer')
        tmp = tiktoken.encoding_for_model(self.model)
        #print('加载tokenizer完毕')
        return tmp

    def encode(self, *args, **kwargs):
        encoder = self.get_encoder()
        return encoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        encoder = self.get_encoder()
        return encoder.decode(*args, **kwargs)



embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "bge-small":"BAAI/bge-small-en"
}