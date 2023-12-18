

import torch.utils.data as Data


class MyDataSet_train(Data.Dataset):
  def __init__(self, x_train1, x_train2, x_label): 
    super(MyDataSet_train, self).__init__()
    self.train_x1 = x_train1
    self.train_x2 = x_train2
    self.train_y = x_label
  
  def __len__(self):
    return self.train_x1.shape[0]
  
  def __getitem__(self, idx):
    return self.train_x1[idx], self.train_x2[idx], self.train_y[idx]


class MyDataSet_test(Data.Dataset):
  def __init__(self, y_test1, y_test2, y_label): 
    super(MyDataSet_test, self).__init__()
    self.test_x1 = y_test1
    self.test_x2 = y_test2
    self.test_y = y_label
  
  def __len__(self):
    return self.test_x1.shape[0]
  
  def __getitem__(self, idx):
    return self.test_x1[idx], self.test_x2[idx], self.test_y[idx]

# class MyDataSet_train(Data.Dataset):
#   def __init__(self, x_train1, adj1, x_train2, x_label): 
#     super(MyDataSet_train, self).__init__()
#     self.train_x1 = x_train1
#     self.adj_matrix1 = adj1
#     self.train_x2 = x_train2
#     self.train_y = x_label
  
#   def __len__(self):
#     return self.train_x1.shape[0]
  
#   def __getitem__(self, idx):
#     return self.train_x1[idx], self.adj_matrix1[idx], self.train_x2[idx], self.train_y[idx]


# class MyDataSet_test(Data.Dataset):
#   def __init__(self, y_test1, adj1, y_test2, y_label): 
#     super(MyDataSet_test, self).__init__()
#     self.test_x1 = y_test1
#     self.adj_matrix1 = adj1
#     self.test_x2 = y_test2
#     self.test_y = y_label
  
#   def __len__(self):
#     return self.test_x1.shape[0]
  
#   def __getitem__(self, idx):
#     return self.test_x1[idx], self.adj_matrix1[idx], self.test_x2[idx], self.test_y[idx]