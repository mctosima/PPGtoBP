"""Initialising Parameters"""
def init_params():

  # Number of features
  seq_len = 1000
  batch_size = 32

  # Params for the generator
  hidden_nodes_g = 50
  layers = 2
  tanh_layer = False

  num_epochs = 20
  learning_rate = 0.0002

  # Params for the Discriminator
  num_cvs = 2
  cv1_out= 10
  cv1_k = 3
  cv1_s = 1
  p1_k = 3
  p1_s = 2
  cv2_out = 10
  cv2_k = 3
  cv2_s = 1
  p2_k = 3
  p2_s = 2
  lambda_cyc =  10.0
  lambda_id = 5.0

  # Create Dictionary - for re-use
  params = {
      'seq_len' : seq_len,
      'batch_size' : batch_size,
      'epochs': num_epochs,
      'learning_rate' : learning_rate,
      'num_cvs' : num_cvs,
      'cv1_out' : cv1_out,
      'cv1_k' : cv1_k,
      'cv1_s' : cv1_s,
      'p1_k' : p1_k,
      'p1_s' : p1_s,
      'cv2_out' : cv2_out,
      'cv2_k' : cv2_k,
      'cv2_s' : cv2_s,
      'p2_k' : p2_k,
      'p2_s' : p2_s,
      'lambda_cyc' : lambda_cyc,
      'lambda_id' : lambda_id
  }

  return params

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def client_update(client_G_AB, client_G_BA, client_D_A, client_D_B,
                  optimizer_G, optimizer_D_A, optimizer_D_B, 
                  train_loader, params, epoch=5):
    """
    This function updates/trains client model on client data
    """
    client_G_AB.train()
    client_G_BA.train()

    client_D_A.train()
    client_D_B.train()
    for e in range(epoch):
        for batch_idx, sample_data in enumerate(train_loader):
          # Set model input
          real_A = Variable(sample_data["A"].type(Tensor))
          real_B = Variable(sample_data["B"].type(Tensor))
          
          # Adversarial GT
          valid = Variable(Tensor(real_A.size(0), 1).fill_(1.0), requires_grad=False)
          fake = Variable(Tensor(real_A.size(0), 1).fill_(0.0), requires_grad=False)
          # ------------------
          #  Train Generators
          # ------------------
          optimizer_G.zero_grad()
          h_g_AB = client_G_AB.init_hidden()
          h_g_BA = client_G_BA.init_hidden()

          # Identity Loss (A,A) and (B,B)
          id_AA = client_G_BA(real_A, h_g_BA)
          id_AA = id_AA.view(params['batch_size'], -1, params['seq_len'])
          id_BB = client_G_AB(real_B, h_g_AB)
          id_BB = id_BB.view(params['batch_size'], -1, params['seq_len'])

          loss_id_A = criterion_identity(id_AA, real_A)
          loss_id_B = criterion_identity(id_BB, real_B)
                      
          loss_identity = (loss_id_A + loss_id_B) / 2

          # GAN loss
          fake_B = client_G_AB(real_A, h_g_AB)
          fake_B = fake_B.view(params['batch_size'], -1, params['seq_len'])
          loss_GAN_AB = criterion_GAN(client_D_B(fake_B), valid)

          fake_A = client_G_BA(real_B, h_g_BA)
          fake_A = fake_A.view(params['batch_size'], -1, params['seq_len'])
          loss_GAN_BA = criterion_GAN(client_D_A(fake_A), valid)

          loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

          # Cycle loss
          recov_A = client_G_BA(fake_B, h_g_BA)
          recov_A = recov_A.view(params['batch_size'], -1, params['seq_len'])
          loss_cycle_A = criterion_cycle(recov_A, real_A)
          recov_B = client_G_AB(fake_A, h_g_AB)
          recov_B = recov_B.view(params['batch_size'], -1, params['seq_len'])
          loss_cycle_B = criterion_cycle(recov_B, real_B)

          loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

          # Total loss
          loss_G = loss_GAN + params['lambda_cyc'] * loss_cycle + params['lambda_id'] * loss_identity

          loss_G.backward()
          optimizer_G.step()

          # -----------------------
          #  Train Discriminator A
          # -----------------------
          optimizer_D_A.zero_grad()

          # Real loss
          loss_real = criterion_GAN(client_D_A(real_A), valid)
          # Fake loss (on batch of previously generated samples)
          fake_A_ = fake_A_buffer.push_and_pop(fake_A)
          loss_fake = criterion_GAN(client_D_A(fake_A_.detach()), fake)
          # Total loss
          loss_D_A = (loss_real + loss_fake) / 2

          loss_D_A.backward()
          optimizer_D_A.step()

          # -----------------------
          #  Train Discriminator B
          # -----------------------
          optimizer_D_B.zero_grad()

          # Real loss
          loss_real = criterion_GAN(client_D_B(real_B), valid)
          # Fake loss (on batch of previously generated samples)
          fake_B_ = fake_B_buffer.push_and_pop(fake_B)
          loss_fake = criterion_GAN(client_D_B(fake_B_.detach()), fake)
          # Total loss
          loss_D_B = (loss_real + loss_fake) / 2

          loss_D_B.backward()
          optimizer_D_B.step()

          loss_D = (loss_D_A + loss_D_B) / 2

    return loss_G.item(), loss_D.item()

def server_aggregate(global_G_AB, global_G_BA, global_D_A, global_D_B, 
                     client_G_AB, client_G_BA, client_D_A, client_D_B):
    ### This will take simple mean of the weights of G_AB ###
      global_dict_G_AB = global_G_AB.state_dict()
      for k in global_dict_G_AB.keys():
          global_dict_G_AB[k] = torch.stack([client_G_AB[i].state_dict()[k].float() for i in range(len(client_G_AB))], 0).mean(0)
      global_G_AB.load_state_dict(global_dict_G_AB)
      for model_G_AB in client_G_AB:
          model_G_AB.load_state_dict(global_G_AB.state_dict())

    ### This will take simple mean of the weights of models ###
      global_dict_G_BA = global_G_BA.state_dict()
      for k in global_dict_G_BA.keys():
          global_dict_G_BA[k] = torch.stack([client_G_BA[i].state_dict()[k].float() for i in range(len(client_G_BA))], 0).mean(0)
      global_G_BA.load_state_dict(global_dict_G_BA)
      for model_G_BA in client_G_BA:
          model_G_BA.load_state_dict(global_G_BA.state_dict())

    ### This will take simple mean of the weights of models ###
      global_dict_D_A = global_D_A.state_dict()
      for k in global_dict_D_A.keys():
          global_dict_D_A[k] = torch.stack([client_D_A[i].state_dict()[k].float() for i in range(len(client_D_A))], 0).mean(0)
      global_D_A.load_state_dict(global_dict_D_A)
      for model_D_A in client_D_A:
          model_D_A.load_state_dict(global_D_A.state_dict())

    ### This will take simple mean of the weights of models ###
      global_dict_D_B = global_D_B.state_dict()
      for k in global_dict_D_B.keys():
          global_dict_D_B[k] = torch.stack([client_D_B[i].state_dict()[k].float() for i in range(len(client_D_B))], 0).mean(0)
      global_D_B.load_state_dict(global_dict_D_B)
      for model_D_B in client_D_B:
          model_D_B.load_state_dict(global_D_B.state_dict())