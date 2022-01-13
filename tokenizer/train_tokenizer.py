from pymongo import MongoClient
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from tqdm import tqdm

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=32000, min_frequency=3, special_tokens=[
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask>"
])

word_tokens = []

client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
codes = client.code_search_net.codes
query = codes.find({'partition': 'train'})


for result in tqdm(query, total=query.count()):
    word_tokens.extend(result['code_tokens'])
    word_tokens.extend(result['docstring_tokens'])

tokenizer.train_from_iterator(word_tokens, trainer=trainer, length=len(word_tokens))

# Save files to disk
tokenizer.save('./vocab/roberta_tokenizer.json', pretty=True)

print(
    tokenizer.encode("<s> hello <unk> Don't you love 🤗 Transformers <mask> yes . </s>").tokens
)