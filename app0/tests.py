from django.test import TestCase

# Create your tests here.

request = '类型001'
encode = request.encode()
print(encode)
print(encode.decode())