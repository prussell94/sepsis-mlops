from kfp import dsl
from kfp import compiler

@dsl.component(base_image='python:3.9-slim')
def hello_op():
    print("Hello from a clean V2 pipeline!")

@dsl.pipeline(name='v2-test-clean')
def my_v2_pipeline():
    hello_op()

if __name__ == '__main__':
    compiler.Compiler().compile(my_v2_pipeline, 'v2_test.yaml')