import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Artifact

# 1. Define the Preprocessing Component
@dsl.container_component
def preprocess_op(
    input_data_path: str,
    output_dataset: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='personal-project-registry/sepsis-preprocess:latest',
        command=['python', 'preprocess.py'],
        args=[
            '--input', input_data_path,
            '--output', output_dataset.path
        ]
    )

# 2. Define the Training Component
@dsl.container_component
def train_op(
    input_dataset: Input[Dataset],
    model_output: Output[Artifact],
    epochs: int = 10,
    batch_size: int = 32
):
    return dsl.ContainerSpec(
        image='personal-project-registry/sepsis-train:latest',
        command=['python', 'train.py'],
        args=[
            '--data', input_dataset.path,
            '--model_save_path', model_output.path,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size)
        ]
    )

# 3. Define the Workflow (The Pipeline)
@dsl.pipeline(
    name='Sepsis Prediction Training Pipeline',
    description='A pipeline that cleans medical data and trains an LSTM model.'
)
def sepsis_pipeline(input_path: str = '/data/raw_sepsis.psv'):
    
    # Task 1: Preprocess
    prep_task = preprocess_op(input_data_path=input_path)
    
    # Task 2: Train (Depends on the output of prep_task)
    train_task = train_op(
        input_dataset=prep_task.outputs['output_dataset'],
        epochs=20,
        batch_size=64
    )

    # Optional: Set resource constraints
    prep_task.set_cpu_limit('2').set_memory_limit('4G')
    train_task.set_gpu_limit('1') # If your cluster has GPUs

if __name__ == '__main__':
    # Compile the pipeline to a YAML file
    kfp.compiler.Compiler().compile(
        pipeline_func=sepsis_pipeline,
        package_path='sepsis_pipeline.yaml'
    )