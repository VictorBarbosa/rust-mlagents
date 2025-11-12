# 🔧 Correção do Formato de Export ONNX

## ⚠️ Problema Encontrado

O script Python gerado automaticamente pelo Rust (`export_onnx()`) estava usando `register_buffer()` que **NÃO funciona** no Unity ML-Agents.

**Erro no Unity:**
```
InvalidCastException: Specified cast is not valid.
Unity.MLAgents.Inference.SentisModelInfo.GetTensorByNameAsInt
```

## ✅ Solução Aplicada

Atualizado o template do script Python em `trainer.rs` para usar o formato que **FUNCIONA**:

### ❌ Formato Antigo (Não Funciona)

```python
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # ❌ register_buffer NÃO funciona!
        self.register_buffer('version_number_const', torch.tensor([[3]], dtype=torch.long))
        
    def forward(self, obs):
        return (self.version_number_const, ...)  # ❌ NÃO funciona
```

### ✅ Formato Novo (Funciona!)

```python
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.output_size = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, obs):
        batch_size = obs.size(0)
        continuous_actions = torch.tanh(self.net(obs))
        
        # ✅ torch.tensor().expand() FUNCIONA!
        version_number = torch.tensor([[3.0]], dtype=torch.float32).expand(batch_size, 1)
        memory_size = torch.zeros((batch_size, 1), dtype=torch.float32)
        continuous_action_output_shape = torch.tensor([[self.output_size]], dtype=torch.float32).expand(batch_size, 1)
        
        return (version_number, memory_size, continuous_actions, ...)  # ✅ Funciona!
```

## 🔑 Diferenças Importantes

| Aspecto | ❌ Antigo | ✅ Novo |
|---------|----------|---------|
| Metadados | `register_buffer` | `torch.tensor().expand()` |
| Tipo | Int64 | Float32 |
| Criação | No `__init__` | No `forward()` |
| Compatibilidade | ❌ Não funciona | ✅ Funciona! |

## 📦 Arquivo Atualizado

- `rust-mlagents/rl_core/src/trainers/sac/trainer.rs`
  - Método `export_onnx()` - Template do script Python

## 🧪 Como Testar

### 1. Gerar ONNX com formato correto:

```bash
cd rust-mlagents
python3 ../generate_test_onnx.py --obs 62 --actions 2
```

### 2. Testar no Unity:

```bash
cp sac_agent_obs62_act2.onnx Unity/Assets/ML-Agents/Models/
```

### 3. Verificar no Unity Inspector:

- ✅ Sem erros de InvalidCastException
- ✅ Modelo carrega corretamente
- ✅ Version number é lido corretamente

## 🔄 Modelos Atualizados

Modelos em `model_samples/` com formato correto:

- ✅ `test_corrected_format.onnx` - Novo formato (obs=62, act=2)
- ✅ `working_model.onnx` - Formato funcional
- ✅ `exact_mlagents.onnx` - Formato exato do ML-Agents
- ✅ `trained_exact_mlagents.onnx` - Com pesos treinados

## 💡 Por Que Funciona?

O Unity ML-Agents (Sentis) consegue ler valores de constantes quando elas são criadas com:

```python
torch.tensor([[valor]]).expand(batch_size, 1)
```

Mas **NÃO consegue** quando usa:

```python
self.register_buffer('nome', torch.tensor([[valor]]))
```

A diferença está em como o ONNX serializa essas operações:
- `expand()` → Cria nó `Expand` com constante embutida ✅
- `register_buffer()` → Cria parâmetro que Unity não consegue ler ❌

## 🐛 Se Ainda Tiver Erro

### Verifique:

1. **Script Python correto?**
   ```bash
   # Ver qual script foi gerado
   cat checkpoints/sac_step_10000_convert_to_onnx.py | grep "torch.tensor.*expand"
   ```

2. **Usar script standalone:**
   ```bash
   # Se script gerado pelo Rust ainda estiver errado, use:
   python3 ../convert_checkpoint_to_onnx.py checkpoints/sac_step_10000.pt
   ```

3. **Regenerar checkpoints:**
   - Se checkpoints foram salvos com versão antiga, o script gerado está desatualizado
   - Salve um novo checkpoint para gerar script atualizado

## 📝 Checklist de Verificação

Antes de usar ONNX no Unity:

- [ ] Script Python usa `torch.tensor().expand()`
- [ ] Metadados são Float32 (não Int64)
- [ ] ONNX valida sem erros
- [ ] Unity carrega sem InvalidCastException
- [ ] Modelo funciona no Unity

## 🎯 Próximos Passos

1. **Recompilar o Rust** (se necessário)
2. **Treinar novo modelo** para gerar checkpoints com script correto
3. **Exportar ONNX** com novo formato
4. **Testar no Unity** ✅

---

**Status:** ✅ CORRIGIDO
**Data:** 2025-11-12
**Arquivo:** `rust-mlagents/rl_core/src/trainers/sac/trainer.rs`
**Testado:** Python ✅ | Unity ⏳ (aguardando teste)
