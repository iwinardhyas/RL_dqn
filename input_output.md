Deep Q-Network (DQN) adalah metode reinforcement learning yang menggabungkan Q-learning dengan deep learning. Dalam use case penjadwalan pekerjaan (job scheduling), DQN dapat digunakan untuk menentukan tindakan (aksi) optimal berdasarkan keadaan (state) sistem saat ini untuk memaksimalkan beberapa metrik kinerja seperti throughput, waktu penyelesaian, atau pemanfaatan sumber daya.

Berikut adalah penjelasan mengenai input dan output pada Q-network dalam konteks DQN untuk kasus penjadwalan pekerjaan:

## Use Case: Job Scheduling

Penjadwalan pekerjaan melibatkan alokasi pekerjaan ke sumber daya (seperti CPU, memori, atau mesin) dalam sistem untuk mencapai tujuan tertentu. Dalam reinforcement learning, agen belajar untuk memilih aksi (tindakan penjadwalan) berdasarkan keadaan (state) sistem saat ini untuk memaksimalkan reward (hasil yang diinginkan).

### 1. Input pada Q-Network

Input untuk Q-Network dalam DQN adalah representasi keadaan sistem saat ini, yang disebut sebagai **state**. Dalam konteks penjadwalan pekerjaan, state harus mencerminkan informasi yang relevan tentang pekerjaan yang perlu dijadwalkan dan status sumber daya yang tersedia.

#### Contoh Detail State:

1. **Informasi Pekerjaan (Jobs Information)**:
   - **Daftar Pekerjaan yang Menunggu**: Informasi mengenai pekerjaan yang belum dijadwalkan seperti:
     - Ukuran pekerjaan (misalnya, durasi atau jumlah instruksi).
     - Prioritas pekerjaan.
     - Deadline pekerjaan.
   - **Atribut Pekerjaan**: Setiap pekerjaan dapat memiliki fitur spesifik seperti:
     - Resource requirements (CPU, memori, dll.).
     - Time constraints.
     - Dependencies (pekerjaan yang harus diselesaikan terlebih dahulu).

2. **Status Sumber Daya (Resources Status)**:
   - **Kapasitas Sumber Daya yang Tersedia**: Jumlah CPU, memori, atau sumber daya lainnya yang tersedia pada waktu tertentu.
   - **Penggunaan Sumber Daya Saat Ini**: Berapa banyak sumber daya yang sudah digunakan oleh pekerjaan yang sedang berjalan.
   - **Status Mesin**: Jika menggunakan banyak mesin, status masing-masing mesin seperti idle atau busy.

3. **Kondisi Sistem Global**:
   - **Waktu Saat Ini**: Posisi waktu saat ini dalam siklus penjadwalan.
   - **Statistik Historis**: Informasi seperti jumlah pekerjaan yang sudah diselesaikan, waktu penyelesaian rata-rata, atau throughput sistem.

#### Representasi State:

State dalam penjadwalan pekerjaan biasanya direpresentasikan sebagai vektor fitur atau tensor yang menggabungkan semua informasi di atas. Contohnya, sebuah state dapat direpresentasikan sebagai vektor berisi informasi seperti:

```
State = [J1_size, J1_priority, J1_deadline, ..., Jn_size, Jn_priority, Jn_deadline, R1_available, R1_used, ..., Rm_available, Rm_used, current_time, ...]
```

### 2. Output pada Q-Network

Output dari Q-Network adalah nilai Q (Q-value) untuk setiap tindakan (aksi) yang mungkin dalam state saat ini. Dalam kasus penjadwalan pekerjaan, tindakan biasanya berhubungan dengan keputusan penjadwalan spesifik seperti alokasi pekerjaan ke sumber daya tertentu atau pengurutan pekerjaan.

#### Contoh Detail Aksi:

1. **Menjadwalkan Pekerjaan ke Sumber Daya**:
   - Memilih pekerjaan mana yang akan dijadwalkan ke sumber daya yang tersedia.
   - Contoh: "Jadwalkan Pekerjaan 3 ke CPU 2".

2. **Pengurutan Pekerjaan**:
   - Menentukan urutan eksekusi pekerjaan yang akan datang.
   - Contoh: "Jalankan Pekerjaan A sebelum Pekerjaan B".

3. **Menentukan Waktu Mulai Pekerjaan**:
   - Memutuskan kapan pekerjaan tertentu harus mulai dieksekusi.
   - Contoh: "Mulai Pekerjaan 4 pada waktu t+5".

#### Representasi Output:

Output dari Q-network biasanya merupakan vektor yang berisi nilai Q untuk setiap tindakan yang mungkin dalam state saat ini. Jika ada \(N\) tindakan yang mungkin, output adalah vektor \(N\)-dimensi:

```
Output = [Q(S, A1), Q(S, A2), ..., Q(S, AN)]
```

Di mana \(Q(S, Ai)\) adalah nilai Q untuk state \(S\) dan aksi \(Ai\).

### 3. Proses Pemilihan Aksi

Setelah Q-Network memberikan output nilai Q untuk semua tindakan yang mungkin, agen memilih tindakan berdasarkan strategi tertentu. Strategi yang umum digunakan adalah **epsilon-greedy**:
- **Epsilon-Greedy**: Dengan probabilitas \(\epsilon\), agen memilih aksi secara acak (eksplorasi); dengan probabilitas \(1 - \epsilon\), agen memilih aksi dengan nilai Q tertinggi (eksploitasi).

### Diagram Proses

```plaintext
+-----------------+             +-------------------+              +--------------+
|  Current State  |  -------->  |   Q-Network (NN)  |  --------->  |  Q-Values    |
|  (System Info)  |             |   (Evaluates Q)   |              |  (For Actions)|
+-----------------+             +-------------------+              +--------------+
        |                           |
        |                           V
        |             +---------------------+
        |             |   Action Selection  |
        |             | (e.g., Epsilon-Greedy)|
        |             +---------------------+
        |                           |
        |                           V
        |             +---------------------+
        |             |    Execute Action   |
        |             | (e.g., Schedule Job)|
        |             +---------------------+
        |                           |
        |                           V
        |             +---------------------+
        |             | Receive Reward and |
        |             | Observe New State  |
        |             +---------------------+
        |                           |
        V                           V
+----------------+        +---------------------+
|  Update State  |        |  Train Q-Network    |
|   Information  |        |  (Using New Data)   |
+----------------+        +---------------------+
```

## Implementasi Contoh dengan DQN

Untuk memberikan gambaran lebih konkret, berikut adalah pseudocode sederhana yang menggambarkan penggunaan DQN untuk penjadwalan pekerjaan:

### Pseudocode

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Definisikan Q-Network sebagai model neural network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
state_size = 20  # Contoh: vektor state berukuran 20 dimensi
action_size = 10  # Contoh: 10 tindakan yang mungkin
learning_rate = 0.001
gamma = 0.99  # Diskon faktor untuk reward masa depan
epsilon = 1.0  # Faktor eksplorasi awal
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 10000

# Inisialisasi Q-Network dan Optimizer
q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Replay buffer untuk menyimpan pengalaman
memory = deque(maxlen=memory_size)

# Fungsi untuk memilih aksi berdasarkan epsilon-greedy policy
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.randint(action_size)  # Eksplorasi: pilih aksi acak
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = q_network(state_tensor).detach().numpy()
    return np.argmax(q_values)  # Eksploitasi: pilih aksi dengan Q-value tertinggi

# Fungsi untuk melatih Q-Network
def train_dqn():
    if len(memory) < batch_size:
        return
    
    # Sampling pengalaman dari replay buffer
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.int64)
        done_tensor = torch.tensor(done, dtype=torch.float32)
        
        # Target Q-value
        with torch.no_grad():
            q_next = q_network(next_state_tensor).max().item()
            q_target = reward_tensor + (1.0 - done_tensor) * gamma * q_next
        
        # Prediksi Q-value
        q_pred = q_network(state_tensor)[action_tensor]
        
        # Hasilkan loss dan optimasi Q-Network
        loss = nn.MSELoss()(q_pred, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Simulasi pelatihan agen DQN
for episode in range(1000):  # Gantilah dengan

 kriteria penghentian yang sesuai
    state = env.reset()  # Inisialisasi state dari lingkungan
    done = False
    
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        # Simpan pengalaman ke replay buffer
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        
        # Latih Q-Network
        train_dqn()
        
    # Kurangi epsilon untuk mengurangi eksplorasi seiring waktu
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Pelatihan selesai!")
```

### Penjelasan Pseudocode:

1. **Q-Network**: Model neural network yang memprediksi nilai Q untuk setiap tindakan berdasarkan input state.
2. **Replay Buffer**: Buffer yang menyimpan pengalaman untuk pelatihan batch yang lebih stabil.
3. **Select Action**: Fungsi untuk memilih aksi berdasarkan kebijakan epsilon-greedy.
4. **Train DQN**: Fungsi untuk melatih Q-Network menggunakan pengalaman yang tersimpan dalam replay buffer.
5. **Simulasi Pelatihan**: Loop yang menjalankan simulasi pelatihan agen, termasuk seleksi aksi, penyimpanan pengalaman, dan pelatihan Q-Network.

Dengan memahami dan mengimplementasikan konsep di atas, Anda dapat mengembangkan sistem DQN untuk memecahkan masalah penjadwalan pekerjaan secara efisien.