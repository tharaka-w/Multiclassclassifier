import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_data(data_dir, anno_csv, feature_length):
    """Return data X and label Y from the annotation csvs or raw files"""
    
    with open(anno_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        first_column_values = []
        second_column_values = []
        
        for row in csv_reader:
            first_column_values.append(row[0])  # File names
            second_column_values.append(row[1])  # Labels

    # Removing header if present
    first_column_values = first_column_values[1:]
    second_column_values = second_column_values[1:]

    X, Y = [], []
    missing_files = []  # Track missing files for debugging

    for index, file_name in enumerate(first_column_values):
        file_path = os.path.join(data_dir, file_name)  # Ensure correct file path

        if os.path.exists(file_path):  # Check if file exists before loading
            x = np.load(file_path)
            y = int(second_column_values[index])

            X.append(x.tolist())  # Convert NumPy array to list
            Y.append(y)
        else:
            missing_files.append(file_name)  # Log missing files

    if missing_files:
        print(f"Warning: {len(missing_files)} files are missing. Check dataset integrity.")

   # return torch.tensor(X_images), torch.tensor(labels_numpy,  dtype=torch.long)
    return torch.tensor(X ,  dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

def softmax(logits):
    return torch.nn.functional.softmax(logits, dim=1)

def pred(X,Theta):
    '''predict Y from X and Theta'''
    '''Return Y'''
    ones = torch.ones((X.shape[0],1))
    X_new = torch.cat((X,ones),dim=1)

    logits = X_new @ Theta  # Compute raw scores
    softmax_probs = softmax(logits)  # Call the softmax function

    pred_Y = torch.argmax(softmax_probs, dim=1) + 1  # Get class index (1-based)
  #  X_test = X_new @ Theta
   # pred_Y = torch.argmax(X_test, dim=1)+1
    return pred_Y

def train_mine():
    train_dir = '/content/data_progS24/data_progS24/train_processed/' # specify your training data directory
    train_anno_file = '/content/data_progS24/data_progS24/labels/train_anno.csv' # specify your training data label directory
    test_dir = '/content/data_progS24/data_progS24/test_processed/' # specify your test data directory
    test_anno_file = '/content/data_progS24/data_progS24/labels/test_anno.csv' # specify your test label directory
    feature_length = 784
    total_class = 5

    #Hyperparameters
    lambda1 = 0.0001
    mini_batch_size = 3100
    initiallearningrate = 0.3
    max_epoch = 250
    
    # Specifying the device to GPU/CPU. Here, 'cuda' means using 'gpu' and 'cpu' means using cpu only
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = np.arange(0, max_epoch)

    #Initialize Theta Matrix W
    W = np.random.rand(feature_length+1, total_class)

    #Read the data and labels from the training data
    X, gt_Y = get_data(data_dir=train_dir, anno_csv=train_anno_file, feature_length=feature_length)
    #Read the data and labels from the testing data
    X_test, gt_Y_test = get_data(data_dir=test_dir, anno_csv=test_anno_file, feature_length=feature_length)

    #adding a 1s column
    ones = torch.ones((X.shape[0],1))
    X_new = torch.cat((X,ones),dim=1)
    X_train = X_new.T

    #One hot Encoding Y
    Y_one_hot = torch.zeros((X.shape[0], total_class))
    Y_one_hot[torch.arange(X.shape[0]), gt_Y-1] = 1

    #Initializing weights
    theta = torch.randn((feature_length +1 ,total_class), dtype=torch.float32)
    theta[:feature_length] *= 0.0001
    
    theta[feature_length] = 1

    loss_vector = []
    accuracy_vector=[]
    Total_error_rate_vector=[]
    accuracy_vector_per_class=torch.zeros((5,len(epochs)))
    error_vector_per_class=torch.zeros((5,len(epochs)))

    

    for epoch in epochs:

        '''Compute the loss, gradient of loss with respect to W,
        and update W with gradient descent/stochastic gradient descent method.
        Observe the loss for each epoch, it should decrease with each epoch.'''
        #Learning rate reducing with epoch
        learningrate = initiallearningrate / (1 + 0.01 * epoch)
        num_batches = X.shape[0] // mini_batch_size

        #Adding random choices to the dataset by shuffling
        indices = torch.randperm(X.shape[0])
        X_new = X_new[indices]  # Shuffle data
        Y_one_hot = Y_one_hot[indices]  # Shuffle labels


        total_loss = 0.0
        total_samples = 0
        for j in range(0, X.shape[0], mini_batch_size):
          #minibatching
          X_batch = X_new[j:j+mini_batch_size]
          Y_batch = Y_one_hot[j:j+mini_batch_size]
          current_batch_size = X_batch.shape[0]

          #Calculations in Theory
          logits = X_batch @ theta
          logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # Prevent overflow
          exp_vals = torch.exp(logits)
          probs = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
          
          #Loss and gradient calculation and L2 regularization gradient calculation

          Loss_per_class =  - torch.sum((Y_batch*torch.log(probs)))/current_batch_size
          Loss = torch.sum(Loss_per_class).item()
          gradient = - (X_batch.T @ (Y_batch - probs))/current_batch_size
          L2_regularized_gradient = 2*lambda1*theta
          L2_regularized_gradient[-1, :] = 0  # to avoid regularizing bias term

          batch_loss = - torch.sum(Y_batch * torch.log(probs)) / current_batch_size
          total_loss += batch_loss.item() * current_batch_size  # Weight by batch size
          total_samples += current_batch_size
          
          #theta update with L2 regularization and gradient
          theta = theta - learningrate*(gradient + L2_regularized_gradient)

        avg_loss2 = total_loss / total_samples
        print(epoch,"loss,"" ",avg_loss2)
        loss_vector.append(avg_loss2)

        #predictions for train
        pred_Y_train = pred(X,theta)

        #for test data calculate error rate
        pred_Y_test = pred(X_test,theta)
        for error in range(5):
          Correct_predictions_per_class = torch.sum(pred_Y_test[gt_Y_test == error+1] == error + 1).item()
          total_samples_per_class = torch.sum(gt_Y_test == error + 1).item()
          accuracy_per_class = (Correct_predictions_per_class / total_samples_per_class)
          error_rate_per_class = 1 - accuracy_per_class
          error_vector_per_class[error,epoch]=error_rate_per_class
        
        #for training data  Calculate accuracy,
        Correct_predictions = torch.sum(pred_Y_train == gt_Y).item()
        total_samples = gt_Y.shape[0]
        accuracy = (Correct_predictions / total_samples) * 100
        Total_Error_rate = 100-accuracy
        accuracy_vector.append(accuracy)
        Total_error_rate_vector.append(Total_Error_rate)
        print(f"Epoch {epoch + 1}: Training Accuracy {accuracy:.2f}%")

        #Accuracy per class calculations
        for t in range(5):
          Correct_predictions_per_class = torch.sum(pred_Y_train[gt_Y == t+1] == t + 1).item()
          total_samples_per_class = torch.sum(gt_Y == t + 1).item()
          accuracy_per_class = (Correct_predictions_per_class / total_samples_per_class) * 100
          accuracy_vector_per_class[t,epoch]=accuracy_per_class

    
    #Plotting curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_vector) + 1), loss_vector, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Training accuracy per class plotting
    for t in range(5):
      plt.figure(figsize=(8, 5))
      plt.plot(range(1, len(accuracy_vector_per_class[t]) + 1), accuracy_vector_per_class[t], marker='o', linestyle='-', color='b', label='Training Accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('accuracy')
      plt.title(f'Training Accuracy Curve for Class {t+1}')
      plt.legend()
      plt.grid(True)
      plt.show()
    #Testing error per class plotting
    for t in range(5):
      plt.figure(figsize=(8, 5))
      plt.plot(range(1, len(error_vector_per_class[t]) + 1), error_vector_per_class[t], marker='o', linestyle='-', color='b', label='Testing Error ')
      plt.xlabel('Epochs')
      plt.ylabel('Error Rate')
      plt.title(f'Testing error Curve for Class {t+1}')
      plt.legend()
      plt.grid(True)
      plt.show()

    #Total testing error plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(Total_error_rate_vector) + 1), Total_error_rate_vector, marker='o', linestyle='-', color='b', label='Testing Error rate')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.title('Total Testing error rate Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Total Training accuracy plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracy_vector) + 1), accuracy_vector, marker='o', linestyle='-', color='b', label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Total Training accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Save the converged Theta (W)
    W = theta
    filehandler = open("multiclass_parameters.txt", "wb")
    pickle.dump(W, filehandler)
    filehandler.close()

    #Trained Digits plotting
    for w in range(5):
      digit = W[:,w]
      digit2=torch.reshape(digit[:-1],(28,28))
      plt.imshow(digit2)
      plt.colorbar()
      plt.title(f'Digit {W + 1}')
     # plt.savefig(f'digit_{wi + 1}.png')
      plt.show()



def test_mine():
    test_dir = '/content/data_progS24/data_progS24/test_processed/' # specify your test data directory
    test_anno_file = '/content/data_progS24/data_progS24/labels/test_anno.csv' # specify your test label directory
    feature_length = 784
    # Specifying the device to GPU/CPU. Here, GPU means 'cuda' and CPU means 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load the Weight Matrix that has been saved after training
    filehandler = open("multiclass_parameters.txt", "rb")
    W = pickle.load(filehandler)

    # Read the data and labels from the testing data
    X, gt_Y = get_data(data_dir=test_dir, anno_csv=test_anno_file, feature_length=feature_length)

    # Predict Y using X and updated W.
    pred_Y = pred(X,W)

    # Calculate accuracy,
    correct_predictions = torch.sum(pred_Y == gt_Y).item()
    total_samples = gt_Y.shape[0]
    accuracy = (correct_predictions / total_samples) * 100

    cm = confusion_matrix(gt_Y.cpu().numpy(), pred_Y.cpu().numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,6), yticklabels=range(1,6))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train_mine()
    test_mine()





