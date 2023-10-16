import numpy as np

class FlashAttnetion:
    def __init__(self, Q=None, K=None, V=None, N=8, d=4,M = 32):
        if Q is None:
            Q = np.random.rand(N, d)
        if K is None:
            K = np.random.rand(N, d)
        if V is None:
            V = np.random.rand(N, d)
        self.query = Q
        self.key = K
        self.value = V
        self.memory = M
        self.dim = d
        self.length = N
    
    def slice(self):
        self.column_block = self.memory // (4*self.dim)
        self.row_block = min(self.column_block, self.dim)
    
    def initialize(self):
        self.output = np.zeros((self.length, self.dim))
        self.ell = np.zeros(self.length)
        self.maximum_row = np.full(self.length, -np.inf)
        
    def run(self, cache = False):
        self.slice()
        self.initialize()
        N = self.length
        
        Q_blocks = np.array_split(self.query, N // self.row_block)  # np.array_split returns List of array
        K_blocks = np.array_split(self.key, N // self.column_block)
        V_blocks = np.array_split(self.value, N // self.column_block)
        
        T_r = len(Q_blocks)
        T_c = len(K_blocks) 
        
        O_tiles = np.array_split(self.output, T_r)
        l_tiles = np.array_split(self.ell, T_r)
        m_tiles = np.array_split(self.maximum_row, T_r)
        if cache:
            log_o = []
        # 5. Loop in Tc
        for j in range(T_c):
            # 6. Load K_j, V_j from HBM to SRAM
            K_j, V_j = K_blocks[j], V_blocks[j]

            # 7. Loop in Tr
            for i in range(T_r):
                # 8. Load Q 𝑖, O 𝑖, ℓ 𝑖, 𝑚 𝑖 from HBM to on-chip SRAM.
                # Q_i, O_i, l_i, m_i = Q_blocks[i], O_tiles[i], l_tiles[i], m_tiles[i]
                Q_i, O_i, l_i, m_i = Q_blocks[i], O_tiles[i], l_tiles[i], m_tiles[i]

                # 9. Compute Sij = Q_iK^T_j which return Br X Bc
                S_ij = np.dot(Q_i, K_j.T)  # Orginally on SRAM
                # Proof of shape
                # print(f"S_ij.shape:{S_ij.shape} vs B_r X B_c : {B_r,B_c}")

                # 10. Compute i) mhat_ij = rowmax(S_ij) return Br,
                #            ii) P_ij = e^(S_ij - mhat_ij) return B_r X B_c,
                #           iii) lhat_ij = rowsum(P_ij) return B_r
                mhat_ij = np.max(S_ij, axis=1)  # max per column
                # add dimension and broadcas

                P_ij = np.exp(S_ij - mhat_ij[:, np.newaxis])
                lhat_ij = np.sum(P_ij, axis=1)

                # 11. Compute i) mnew_i = max(m_i, mhat_ij) return B_r,
                #            ii) lnew_i = exp(m_i - mnew_i)l_i + exp(mhat_ij-mnew_i)lhat_ij return B_r
                mnew_i = np.maximum(m_i, mhat_ij)
                lnew_i = np.exp(m_i - mnew_i) * l_i + np.exp(mhat_ij - mnew_i) * lhat_ij

                # 12. O_i = diag(lnew_i)^-1 * (diag(l_i)exp(m_i - mnew_i)*O_i + exp(mhat_ij-mnew_i)lhat_ij) return B_r
                # TODO
                O_tiles[i] = (1 / lnew_i)[:, np.newaxis] * (
                    (l_i * np.exp(m_i - mnew_i))[:,np.newaxis] * O_i
                    + (np.exp(mhat_ij - mnew_i)[:,np.newaxis] * P_ij) @ V_j
                )
                if cache:
                    log_o.append(np.concatenate(O_tiles, axis=0))
                # 13. override
                l_tiles[i], m_tiles[i] = lnew_i, mnew_i
            # 14. O = concatenate(O_1, O_2, ... , O_T_r) return N X d
        if cache:
            return np.concatenate(O_tiles, axis=0), log_o
        return np.concatenate(O_tiles, axis=0)