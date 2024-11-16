# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

Please note I wanted to directly query S3 instead of downloading everything from S3, I did it as an execersice on how that aproach can be done

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

 - I chose ResNet18 (pretrained) as it is pretrained on ImageNet, a large-scale dataset suitable for general image classification. ResNet18 is known for its efficiency and effectiveness in various image classification tasks, making it an ideal model for transfer learning in this project.

 **NOTE** I ran the hyperparamenter multiple times with multipless variables (almost every combination) rising multiples times out as at some point I was running max jobs 25, concurrent jobs 10. As such if you are running this project can be expensive

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

To fine-tune the model, I experimented with various hyperparameters: optimizers (Adam and SGD), batch sizes (20, 25, 50, 100), epochs (1, 3, 5, 10), and learning rates (beginning with 0.0001 and adjusting by 25%). This process involved analyzing logs and outcomes, iterating on parameter values, and assessing how changes affected performance metrics. The optimal hyperparameters identified were: Adam optimizer, a learning rate of 0.0001, batch size of 50, and 5 epochs.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

The profiling and debugging process in SageMaker helped identify key performance bottlenecks and errors such as PoorWeightInitialization. Through these insights, I iteratively adjusted the hyperparameters, improving training efficiency and overall model accuracy. For example, I could optimize the initialization and tune learning rate and batch size based on profiling data to avoid common pitfalls in training stability.

**TODO** Remember to provide the profiler html/pdf file in your submission.
Done

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The model was deployed to a SageMaker endpoint, providing real-time predictions. To query the endpoint, you can use SageMaker’s predictor.predict() method, ensuring that your input image is preprocessed to match the model’s input requirements (e.g., resizing to 224x224 and normalizing)


**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
Done

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
