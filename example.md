Untuk memahami bagaimana Reinforcement Learning (RL) dengan Deep Q-Network (DQN) diterapkan pada kasus penggunaan penjadwalan pekerjaan (job scheduling), mari kita bahas langkah demi langkah prosesnya. Penjadwalan pekerjaan sering melibatkan pengalokasian pekerjaan ke sumber daya tertentu (misalnya, mesin, server) untuk meminimalkan waktu penyelesaian (makespan), mengoptimalkan utilisasi, atau memenuhi tenggat waktu.

### Langkah-langkah Proses DQN pada Job Scheduling

#### 1. **Definisikan Lingkungan (Environment) dan Agen**

   - **Lingkungan (Environment):** Ini mencakup sistem penjadwalan pekerjaan yang ingin kita optimalkan. Lingkungan mencakup pekerjaan yang harus dijadwalkan, sumber daya yang tersedia, dan aturan atau batasan penjadwalan.
   - **Agen:** Agen adalah entitas yang membuat keputusan penjadwalan berdasarkan keadaan (state) lingkungan dan belajar dari hasil tindakannya (action).

#### 2. **Definisikan State, Action, dan Reward**

   - **State (S):** Keadaan lingkungan saat ini. Misalnya:
     - Daftar pekerjaan yang sedang menunggu.
     - Status saat ini dari setiap sumber daya (misalnya, apakah sedang sibuk atau idle).
     - Waktu yang telah berlalu dan deadline yang tersisa.
   
   - **Action (A):** Tindakan yang dapat diambil oleh agen. Misalnya:
     - Menjadwalkan pekerjaan tertentu ke sumber daya tertentu.
     - Menunda pekerjaan atau memprioritaskan pekerjaan lain.
   
   - **Reward (R):** Umpan balik dari lingkungan setelah agen mengambil tindakan. Reward dirancang untuk mendorong agen melakukan tindakan yang mengarah pada tujuan sistem penjadwalan. Misalnya:
     - Memberikan reward positif untuk menyelesaikan pekerjaan tepat waktu.
     - Memberikan penalti untuk waktu idle sumber daya atau keterlambatan dalam penyelesaian pekerjaan.

#### 3. **Bangun Model Q-Network**

   - **Q-Network:** Jaringan saraf tiruan yang digunakan untuk memprediksi nilai Q, yaitu nilai dari pasangan keadaan dan tindakan. Model ini mengambil state sebagai input dan memberikan nilai Q untuk setiap possible action.
   - **Parameter:** Jaringan dilatih menggunakan parameter seperti bobot dan bias yang diperbarui melalui proses backpropagation.

#### 4. **Proses Pelatihan DQN**

   - **Inisialisasi:** Mulai dengan menginisialisasi replay memory untuk menyimpan pengalaman (state, action, reward, next state) dan menetapkan Q-Network dengan parameter acak.
   - **Loop Pelatihan:** Untuk setiap episode pelatihan:
     - **Start State:** Tentukan state awal dari lingkungan.
     - **Action Selection (ε-greedy):** Pilih tindakan berdasarkan trade-off antara eksplorasi dan eksploitasi. Dengan probabilitas ε, pilih tindakan acak (eksplorasi); sebaliknya, pilih tindakan dengan nilai Q tertinggi (eksploitasi).
     - **Take Action:** Ambil tindakan dan amati reward dan next state dari lingkungan.
     - **Store Experience:** Simpan tuple pengalaman (state, action, reward, next state) ke dalam replay memory.
     - **Sample from Memory:** Ambil sampel batch kecil dari pengalaman dalam replay memory untuk pelatihan.
     - **Update Q-Value:** Perbarui nilai Q menggunakan fungsi kehilangan yang meminimalkan perbedaan antara nilai Q prediksi dan target (y):
       \[
       y = r + \gamma \cdot \max_{a'} Q'(s', a')
       \]
       di mana \(Q'\) adalah nilai Q target, \(r\) adalah reward, \(s'\) adalah next state, dan \(\gamma\) adalah faktor diskon.
     - **Train Q-Network:** Lakukan backpropagation untuk memperbarui parameter jaringan Q-Network.

#### 5. **Evaluasi Kinerja**

   - Setelah beberapa episode pelatihan, evaluasi kinerja agen dengan memantau metrik kunci seperti cumulative reward, makespan, resource utilization, dan task completion rate.
   - **Convergence:** Periksa apakah cumulative reward atau nilai Q stabil, menunjukkan bahwa agen telah belajar kebijakan yang optimal.

### Step by Step Execution

Mari kita lihat langkah-langkah di atas dalam konteks penerapan nyata DQN untuk penjadwalan pekerjaan:

#### Langkah 1: Definisikan Lingkungan dan Agen

```python
class JobSchedulingEnv:
    def __init__(self, jobs, resources):
        self.jobs = jobs
        self.resources = resources
        self.state = self._initialize_state()
    
    def _initialize_state(self):
        # Inisialisasi state awal, misalnya daftar pekerjaan dan status sumber daya
        return {"jobs": self.jobs, "resources": self.resources}
    
    def step(self, action):
        # Eksekusi tindakan (action) dan kembalikan (next_state, reward, done, info)
        next_state = self._get_next_state(action)
        reward = self._compute_reward(action)
        done = self._check_done()
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.state = self._initialize_state()
        return self.state

    def _get_next_state(self, action):
        # Logika untuk memperbarui state berdasarkan tindakan yang diambil
        pass
    
    def _compute_reward(self, action):
        # Logika untuk menghitung reward
        pass
    
    def _check_done(self):
        # Logika untuk memeriksa apakah episode sudah selesai
        pass
```

#### Langkah 2: Definisikan State, Action, dan Reward

Misalnya, kita memiliki pekerjaan dengan atribut seperti waktu penyelesaian dan deadline. Sumber daya mungkin memiliki atribut seperti kapasitas dan status (idle atau sibuk).

- **State:** Representasi pekerjaan yang menunggu dan status sumber daya.
- **Action:** Menjadwalkan pekerjaan tertentu ke sumber daya tertentu.
- **Reward:** Positif jika pekerjaan diselesaikan tepat waktu, negatif jika melewati deadline.

#### Langkah 3: Bangun Model Q-Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

- **Input Dim:** Dimensi dari state yang dikodekan (misalnya, status pekerjaan dan sumber daya).
- **Output Dim:** Dimensi dari tindakan yang mungkin (misalnya, kombinasi pekerjaan dan sumber daya).

#### Langkah 4: Proses Pelatihan DQN

```python
import numpy as np
from collections import deque

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
memory = deque(maxlen=2000)

# Inisialisasi lingkungan dan Q-Network
env = JobSchedulingEnv(jobs, resources)
q_network = QNetwork(input_dim, output_dim)
target_network = QNetwork(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for episode in range(1000):  # Jumlah episode pelatihan
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Ekstraksi state menjadi tensor untuk input ke Q-Network
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        if np.random.rand() <= epsilon:
            # Pilih tindakan secara acak (eksplorasi)
            action = np.random.choice(output_dim)
        else:
            # Pilih tindakan dengan nilai Q tertinggi (eksploitasi)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Simpan pengalaman ke replay memory
        memory.append((state, action, reward, next_state, done))
        
        if len(memory) >= batch_size:
            # Sampling pengalaman untuk pelatihan
            batch = random.sample(memory, batch_size)
            state_batch = torch.tensor([exp[0] for exp in batch], dtype=torch.float32)
            action_batch = torch.tensor([exp[1] for exp in batch])
            reward_batch = torch.tensor([exp[2] for exp in batch], dtype=torch.float32)
            next_state_batch = torch.tensor([exp[3] for exp in batch], dtype=torch.float32)
            done_batch = torch.tensor([exp[4] for exp in batch], dtype=torch.float32)
            
            # Hitung nilai Q target
            q_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            next_q_values = target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
            
            # Perbarui Q-Network
            loss = criterion(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step

()
        
        state = next_state
    
    # Update epsilon untuk mengurangi eksplorasi
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Update target network secara periodik
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    print(f"Episode {episode}, Total Reward: {total_reward}")

```

- **Replay Memory:** Menyimpan pengalaman untuk sampling dan pelatihan.
- **Target Network:** Digunakan untuk menstabilkan pelatihan dengan menyediakan target nilai Q yang lebih stabil.
- **Epsilon-greedy:** Strategi untuk memilih antara eksplorasi dan eksploitasi.

#### Langkah 5: Evaluasi Kinerja

Setelah pelatihan, evaluasi agen DQN pada set pekerjaan baru atau simulasi untuk melihat seberapa baik model menyelesaikan penjadwalan dengan tujuan yang diinginkan.

```python
state = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    q_values = q_network(state_tensor)
    action = torch.argmax(q_values).item()
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Total Reward during Evaluation: {total_reward}")
```

- **Total Reward:** Dapat digunakan sebagai ukuran utama untuk mengevaluasi kinerja model selama evaluasi.

### Ringkasan Output pada Setiap Langkah

1. **Inisialisasi:** Lingkungan dan agen DQN siap dengan state dan tindakan yang ditentukan.
2. **Pelatihan:** Proses pelatihan melibatkan pengambilan tindakan, menerima reward, memperbarui nilai Q, dan mengurangi eksplorasi (epsilon).
3. **Evaluasi:** Kinerja agen dievaluasi berdasarkan cumulative reward dan metrik kinerja lainnya dalam konteks penjadwalan pekerjaan.

Melalui langkah-langkah ini, kita dapat menggunakan DQN untuk membangun agen yang mampu mengelola penjadwalan pekerjaan secara efisien dan efektif.