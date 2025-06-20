class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op

	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		 # Implement addition here
		other = other if isinstance(other,Value) else Value(other)
		out = Value( self.data + other.data, (self,other), '+' )

		def _backward():
			self.grad += out.grad
			other.grad += out.grad
		out._backward = _backward

		return out


	def __mul__(self, other):
		# Implement multiplication here
		other = other if isinstance(other,Value) else Value(other)
		out = Value(self.data * other.data, (self,other), '*' )

		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out._backward = _backward

		return out

	def relu(self):
		# Implement ReLU here
		out = Value( max(self.data,0.0), (self,) ,'Relu' )
		def _backward():
			self.grad += out.grad if out.data != 0.0 else 0.0
		out._backward = _backward

		return out

	def backward(self):
		# Implement backward pass here
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)
		self.grad = 1

		for v in reversed(topo):
			v._backward()

a = Value(2)
b = Value(3)
c = Value(10)
d = a + b * c 
e = Value(7) * Value(2)
f = e + d
g = f.relu() 
g.grad = 1
g.backward() 

print(a,b,c,d,e,f,g)

# a = Value(2.0)
# b = Value(-4.0)
# c = Value(3.0)
# d = a*b + c
# d.backward()
# print(a.grad)
