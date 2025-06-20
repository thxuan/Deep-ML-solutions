import numpy as np
np.random.seed(42)
#GELU Activation Function
def gelu(x):
	return 0.5 * x * ( 1 + np.tanh( np.sqrt(2/np.pi)*(x + 0.044715*(x**3)) ) )
# Layer Normalization
def layer_norm(x,g,b):
	epsilon = 1e-8
	return ( ( g * (x - np.mean(x, axis=-1,keepdims=True )) ) / \
		 np.sqrt(np.var(x,axis=-1,keepdims=True) +  epsilon) ) + b
#Feedforward Network
def ffn(x,w1,w2):
	x_proj = x @ w1         
	x_activated = gelu(x_proj)
	x_output = x_activated @ w2 
	return x_output
def softmax(x):
	x = x - np.max(x, axis=-1, keepdims=True)
	return (np.exp(x))/(np.sum( np.exp(x), axis=-1, keepdims=True ))
def multi_head_attention(x,n_head,w_q, w_k, w_v, w_o):
	head = []
	for n in range(n_head):
		Q = x @ w_q[n]
		K = x @ w_k[n]
		V = x @ w_v[n]
		mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
		S = (Q @ K.T)/np.sqrt(5) + mask
		head.append(softmax(S)@V)
	output = np.concatenate(head,axis=-1) @ w_o
	return output
def generate(inputs, n_head, wte, wpe, ffn_weights, att_weights, ln_f):
	x = wte[inputs] + wpe[range(len(inputs))]
	# Transformer Block
	x1 = layer_norm( (x + multi_head_attention(x,n_head,**att_weights)),**ln_f )
	x2 = layer_norm( (x1 + ffn(x1, **ffn_weights)), **ln_f )
	logits = x2[-1] @ wte.T
	return logits

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	encoder, hparams, params = load_encoder_hparams_and_params()
	n_head = hparams["n_head"]
	input_ids = encoder.encode(prompt)
	assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]	
	next_token_id = []
	for _ in range(n_tokens_to_generate):
		logits = generate(input_ids,n_head,**params)
		probs = softmax(logits)
		token_id = int(np.argmax(probs))		
		next_token_id.append(token_id)
		input_ids.append(token_id)	
	return encoder.decode(next_token_id)	

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,
		"n_head": 2 
	}

	params = {
		"wte": np.random.rand(3, 10),    #word token embedding
		"wpe": np.random.rand(1024, 10), #word postion embedding
		"att_weights": {
			"w_q" : [np.zeros( (10, 5)) for _ in range(2)],  # 10 / 5
			"w_k" : [np.zeros((10, 5)) for _ in range(2)],
			"w_v" : [np.zeros((10, 5)) for _ in range(2)],
			"w_o" : np.zeros((10, 10)),
		},
		"ffn_weights":{
			"w1" : np.zeros((10, 40)), 
			"w2" : np.zeros((40, 10)) 	
		},
		"ln_f": {
			"g": np.ones(10),  			#learnable scaling 
			"b": np.zeros(10), 			#bias parameters
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params


#print(gen_text("world", n_tokens_to_generate=3))
#expect output  world world world

print(gen_text("hello", n_tokens_to_generate=5))
#expect output  hello hello hello <UNK> <UNK>

#print(gen_text("hello world", n_tokens_to_generate=10))
#expect output  world world world world world world world world world world