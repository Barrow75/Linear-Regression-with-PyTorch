import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


print(torch.__version__)
print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)


# create, load, and read the data
sample_data ={
"Time Studying (hours)": [1, 2, 4, 5, 6, 7, 9, 13, 14, 15],
    "Exam Grades (%)": [23, 30, 54, 62, 76, 81, 87,95,99, 100]
}

df = pd.DataFrame(sample_data)
df.to_csv("study_score.csv", index=False)
loading_data = pd.read_csv("study_score.csv")

# weights and biases currently unknown
weights = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
bias = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

# extract the data into numpy arrays
x_data = loading_data["Time Studying (hours)"].to_numpy()
y_data = loading_data["Exam Grades (%)"].to_numpy()

# Turn data into tensors
tensorX = torch.tensor(x_data, dtype=torch.float32).unsqueeze(dim=1)
tensorY = torch.tensor(y_data, dtype=torch.float32).unsqueeze(dim=1)

print(tensorX)
print(tensorY)


# linear regression line
slope, intercept = np.polyfit(x_data, y_data, 1)
y_predicted = slope * x_data + intercept

# Calculate the MSE

def MSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse


mse = MSE(y_data, y_predicted)

print(f'MSE: {mse: .2f}')

# Calculate Grad Descent
optimizer = torch.optim.SGD([weights, bias], lr=.01)
loss_fn = torch.nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    y_pred = weights * tensorX + bias

    loss = loss_fn(y_pred, tensorY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item(): .4f}")

print(f"Trained Weights (Slope): {weights.item():.2f}")
print(f"Trained Bias (Intercept): {bias.item():.2f}")


# visualize current data
def visualize(x = tensorX, y = tensorY, weights=None, bias=None, new_hours=None, new_score=None):

    x = x.numpy()
    y = y.numpy()
    plt.scatter(x, y, color='blue', label='Data Points')

    if weights is not None and bias is not None:
        x_range = torch.linspace(x.min(), x.max(), 100).unsqueeze(dim=1)
        y_range = weights * x_range + bias
        plt.plot(x_range.detach().numpy(), y_range.detach().numpy(), color='red', label=f'Regression Line')

    if new_hours is not None and new_score is not None:
        plt.scatter(new_hours, new_score, color='green', label=f'New Prediction', marker='x', s=100)
    plt.title('Hours studied vs Test Score')
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Scores")

    plt.show()
    return y_predicted
y_predicted = visualize()

# predict new data
new_hours = torch.tensor([3.0, 8.0, 10.0], dtype=torch.float32).unsqueeze(dim=1)
new_score = (weights * new_hours +  bias).detach().numpy()
#plt.scatter(new_hours, new_score, color='green', label=f'New Predictions')

visualize(tensorX,tensorY,weights,bias,new_hours=new_hours.numpy(),new_score=new_score)