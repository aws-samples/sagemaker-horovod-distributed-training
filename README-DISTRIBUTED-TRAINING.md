## Sagemaker Distributed Training with Parameter Server and Horovod

### Distributed training with SageMaker's script mode using Parameter Server and Horovod 

This lab demonstrates two concepts on a simple MNIST dataset and a TensorFlow deep learning framework:
- SageMaker distributed training with Parameter Server
- SageMaker distributed training with Horovod 
Both labs use SageMaker's "script mode" which allows model's training script to be used as an entry point to SageMaker API. Thus, one can bring into SageMaker existing model scripts without making any modifications to them.

#### A. SageMaker's Distributed Training Framework based on Parameter Servers

Parameter Server mode allows one to distribute training workload accross multiple compute instances by implementing a parameter server on each of compute instance to handle model's gradient increment averaging and distributing updated model weights. SageMaker implements an asynchronous parameter server mode whereby gradients and model weights of each worker are updated individually, without waiting for the rest of workers to complete their SGD step. One parameter server is instantiated on each compute instance. For example, for train_instance_count = N, you will have total instances = N, ParameterServer = N,Workers = N-1 (one of the instances becomes a Master and does not carry SGD computations). Currenly, there is no provisions to instantiate multiple workers on a compute instance, so there is a limited advantage of selecting compute instances with a large number of compute cores. However, some deep learning frameworks (eg. TensorFlow) have ability to spread their workload accross multiple cores within the same compute instance, in which case selecting a CPU instance with 8+ cores is advisable. Please, note that Parameter Server is implemented on CPU cores, while workers can be implemented on either CPUs or GPUs. 

#### B. SageMaker's Horovod Distributed Training Framework 

What is Horovod? It is a framework allowing a user to distribute a deep learning workload among multiple compute nodes and take advantage of inherent parallelism of deep learning training process. It is available for both CPU and GPU AWS compute instances. Horovod follows the Message Passing Interface (MPI) model in All-Reduce fashion. This is a popular standard for passing messages and managing communication between nodes in a high-performance distributed computing environment. 

#### C. Further improving performance by co-locating compute instances.

To reduce network delays between compute nodes, it is advisable to locate them within the same subnet. We will demonstrate how to deploy subnets withing the same VPC and assign compute instances to to them\

In this lab, we will be instantiating CPU compute nodes for simplicity and scalability. 

#### Steps for launching Jupyter Notebook:
Open SageMaker Console
![Navigate to Sagemaker Service](../images/image-1.PNG "Navigate to Sagemaker")

Navigate to SageMaker Notebooks
![Navigate to Sagemaker Notebooks](../images/image-2.PNG "Navigate to Sagemaker")

Create a SageMaker Notebook Instance
![Create a Sagemaker Notebooks](../images/image-3.PNG "Navigate to Sagemaker")

Give the SageMaker Notebook Instance a name (note that '_' are not allowed) and click on "Create a new role".
![Create a Sagemaker Notebooks](../images/image-4.PNG "Navigate to Sagemaker")





- Navigate the above file structure to the notebook in 'notebooks' directory
- If prompted, select Jupyter kernel conda_tensorflow_p36
- Launch and execute the notebook. 
Note that depending on your choice of the host machine, it may take as long as 10 min to build the container the 1st time out. 

## License

This library is licensed under the Apache 2.0 License. 
