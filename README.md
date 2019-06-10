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

### Steps for launching Jupyter Notebook:
#### Select one of the following AWS Regions in your AWS console:
![Navigate to Sagemaker Service](/images/image-20.png)


#### Open SageMaker Console by clicking on "Services" and searching for Sagemaker
![Navigate to Sagemaker Service](/images/image-1.png)



#### Navigate to SageMaker Notebooks
![Navigate to Sagemaker Notebooks](/images/image-2.png)



#### Create a SageMaker Notebook Instance
![Creae Sagemaker Notebooks](/images/image-3.png)



#### Give the SageMaker Notebook Instance a name (note that '_' are not allowed) and click on "Create a new role".
![Name Sagemaker Notebooks](/images/image-4.png)



#### Select "Any S3 bucket" and click on "Create role"
![Create IAM role for Sagemaker](/images/image-5.png)



#### We now need to add a few more security policies to our newly created IAM SageMaker role.

#### Click on newly created IAM SageMaker role
![Create IAM role for Sagemaker](/images/image-6.png)



#### Click on "Attach Policies" button
#### Search for "EC2Container" and add AmazonEC2ContanerRegistryFullAccess policy (click on the radio button to the left)
![Create IAM role for Sagemaker](/images/image-7.png)



#### Search for "VPC" and add AmazonVPCAccess policy (click on the radio button to the left)
![Create IAM role for Sagemaker](/images/image-8.png)



#### Click on "Attach Policies" button. Your policy list for the SageMaker IAM role should look like this:
![Create IAM role for Sagemaker](/images/image-9.png)



#### We need a custom policy to allow full access to CloudFormation service. 
"Add in-line policy". Select JSON tab and paste the following:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": "cloudformation:*",
            "Resource": "*"
        }
    ]
}
```


![Create IAM role for Sagemaker](/images/image-10.png)

#### Give your policy a name and click on "Create Policy"
![Create IAM role for Sagemaker](/images/image-11.png)



#### Your policy list for the SageMaker IAM role should look like this:
![Create IAM role for Sagemaker](/images/image-12.png)



#### Switch to the previous browser tab with SageMaker Notebook, scroll down to "Git Repositories", click on "Repository" and select "Add Repository to Amazon SageMaker"
![Create IAM role for Sagemaker](/images/image-13.png)



#### Select 'GitHub repository icon', give it a name, past repo's URL (https://github.com/aws-samples/sagemaker-horovod-distributed-training) and click on "Add repository"
![Create IAM role for Sagemaker](/images/image-14.png)



#### If successful, you will see the Git repo listed under SageMaker Git Repositories:
![Create IAM role for Sagemaker](/images/image-15.png)


#### You can now close this browser's tab and go back to the previous tab where we were creating the notebook. Click on "refresh" button on Git Repositories pane. The github repo's name should now be available in the drop-down list. Select it and click on "Create Notebook Instance" at the bottom of the page.
![Create IAM role for Sagemaker](/images/image-16.png)



#### You will see your notebook in "Pending" status. It will take a few minutes for the Jupyter Server to start up and clone your repo. When the status changes to "InService", click on "Open Jupyter" next to the status.
![Create IAM role for Sagemaker](/images/image-17.png)

#### You will now see in a separate browser tab a familar Jupyter Notebook interface. Click on the file icon and you will see the repository that was checked out. Navigate into 'notebooks' directory and open two notebooks there:

- tensorflow_script_mode_training_and_serving
- tensorflow_script_mode_horovod
![Create IAM role for Sagemaker](/images/image-18.png)



#### It may take up to a minute for the notebooks to fully load into your browser and for the kernels to start up, so please, be patient.
#### Click 'Cell' -> 'Run All' in both notebooks. Some cells in the notebooks may take up to 15 min to fully execute. Be patient.
#### If prompted, select Jupyter kernel 'conda_tensorflow_p36'
![Create IAM role for Sagemaker](/images/image-19.png)


Note that depending on your choice of the host machine, it may take as long as 10 min to build the container the 1st time out. 

This work is based on two SageMaker examples available in AWS SageMaker Examples directory:
- https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/tensorflow_script_mode_training_and_serving
and
- https://github.com/awslabs/amazon-sagemaker-examples/tree/af765120763364193f099af0b283767cc2228ad3/sagemaker-python-sdk/tensorflow_script_mode_horovod


## License

This library is licensed under the Apache 2.0 License. 
