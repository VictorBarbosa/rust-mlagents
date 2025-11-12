# SAC ONNX Export Tests

Este diretório contém testes unitários para validar o processo completo de treinamento e exportação ONNX do algoritmo SAC.

## Testes Disponíveis

### 1. `test_sac_training_and_export`
Teste completo que:
- ✅ Cria um ambiente simples de treinamento
- ✅ Treina um agente SAC por 10 episódios
- ✅ Salva checkpoint em formato `.pt`
- ✅ Exporta modelo para ONNX
- ✅ Valida geração de todos os arquivos necessários
- ✅ Valida estrutura do script de conversão Python

### 2. `test_onnx_conversion_script_structure`
Teste que valida:
- ✅ Geração correta do script de conversão Python
- ✅ Estrutura do modelo ActorNetwork
- ✅ Nomes corretos dos outputs ONNX
- ✅ Configuração dos dynamic_axes
- ✅ Parâmetros de export (opset_version, do_constant_folding, etc.)

## Como Executar

### Executar todos os testes SAC
```bash
cd rust-mlagents
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
DYLD_LIBRARY_PATH=/opt/homebrew/lib/python3.14/site-packages/torch/lib \
cargo test --package rl_core --lib sac::test_export::tests -- --nocapture
```

### Executar teste específico
```bash
cd rust-mlagents
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
DYLD_LIBRARY_PATH=/opt/homebrew/lib/python3.14/site-packages/torch/lib \
cargo test --package rl_core --lib sac::test_export::tests::test_sac_training_and_export -- --nocapture
```

## Estrutura do Export ONNX

O modelo exportado segue **exatamente** o formato esperado pelo Unity ML-Agents:

### Input
- `vector_observation`: [batch, obs_dim] (float32)

### Outputs
- `version_number`: [batch, 1] (int32) - versão do modelo (3)
- `memory_size`: [batch, 1] (int32) - tamanho da memória (0 para modelos sem memória)
- `continuous_actions`: [batch, action_dim] (float32) - ações contínuas
- `continuous_action_output_shape`: [batch, 1] (int32) - dimensão das ações
- `deterministic_continuous_actions`: [batch, action_dim] (float32) - ações determinísticas

## Arquivos Gerados

Após o treinamento e export, os seguintes arquivos são gerados:

```
checkpoint_dir/
├── checkpoint.pt              # Checkpoint padrão (ML-Agents compatible)
├── test_model.pt              # Checkpoint nomeado
├── test_model_full.pt         # VarStore completo
├── test_model.onnx            # Modelo ONNX (Unity compatible)
├── test_model_convert_to_onnx.py  # Script de conversão
├── test_model_metadata.json   # Metadados do modelo
└── metadata.json              # Metadados de treinamento
```

## Validações Realizadas

Os testes validam:

1. ✅ Criação e inicialização do trainer SAC
2. ✅ Loop de treinamento funcional
3. ✅ Salvamento de checkpoint `.pt`
4. ✅ Geração do arquivo `checkpoint.pt` (padrão ML-Agents)
5. ✅ Geração de arquivo de metadata
6. ✅ Export para ONNX com estrutura correta
7. ✅ Script de conversão Python com:
   - Estrutura correta do modelo ActorNetwork
   - Forward method retornando 5 outputs
   - Configuração ONNX com opset_version=11
   - Dynamic axes para todos inputs/outputs
   - Nomes corretos de inputs/outputs
8. ✅ Arquivo ONNX gerado e acessível
9. ✅ Metadata JSON com informações do modelo

## Notas Importantes

### PyTorch Version Mismatch
O tch-rs 0.18 espera PyTorch 2.5.1, mas pode funcionar com versões mais recentes usando:
```bash
LIBTORCH_BYPASS_VERSION_CHECK=1
```

### Dynamic Library Path
No macOS, é necessário configurar o caminho da biblioteca PyTorch:
```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib/python3.14/site-packages/torch/lib
```

No Linux, use:
```bash
LD_LIBRARY_PATH=/path/to/torch/lib
```

### Conversão Automática
O sistema tenta converter automaticamente o `.pt` para `.onnx` usando Python. Se falhar:
1. O script de conversão fica salvo
2. Pode ser executado manualmente: `python3 test_model_convert_to_onnx.py`

## Compatibilidade

- ✅ Unity ML-Agents (Barracuda)
- ✅ PyTorch 2.5+
- ✅ ONNX Runtime
- ✅ Opset version 11

## Troubleshooting

### Erro: "Cannot find a libtorch install"
```bash
export LIBTORCH_USE_PYTORCH=1
```

### Erro: "Library not loaded: libtorch_cpu.dylib"
```bash
export DYLD_LIBRARY_PATH=/path/to/torch/lib
```

### Erro: "this tch version expects PyTorch X.X.X"
```bash
export LIBTORCH_BYPASS_VERSION_CHECK=1
```
