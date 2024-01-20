### 2023-10-06
- Criamos uma GUI para alteração de voz em tempo real: go-realtime-gui.bat/gui_v1.py (observe que você deve escolher o mesmo tipo de dispositivo de entrada e saída, por exemplo, MME e MME).
- Treinamos um modelo RMVPE de extração de pitch melhor.
- Otimizar o layout da GUI de inferência.

### 2023-08-13
1-Correção de bug regular
- Alterar o número total mínimo de épocas para 1 e alterar o número total mínimo de epoch para 2
- Correção de erros de treinamento por não usar modelos de pré-treinamento
- Após a separação dos vocais de acompanhamento, limpe a memória dos gráficos
- Alterar o caminho absoluto do faiss save para o caminho relativo
- Suporte a caminhos com espaços (tanto o caminho do conjunto de treinamento quanto o nome do experimento são suportados, e os erros não serão mais relatados)
- A lista de arquivos cancela a codificação utf8 obrigatória
- Resolver o problema de consumo de CPU causado pela busca do faiss durante alterações de voz em tempo real

Atualizações do 2-Key
- Treine o modelo de extração de pitch vocal de código aberto mais forte do momento, o RMVPE, e use-o para treinamento de RVC, inferência off-line/em tempo real, com suporte a PyTorch/Onnx/DirectML
- Suporte para placas gráficas AMD e Intel por meio do Pytorch_DML

(1) Mudança de voz em tempo real (2) Inferência (3) Separação do acompanhamento vocal (4) Não há suporte para treinamento no momento, mudaremos para treinamento de CPU; há suporte para inferência RMVPE de gpu por Onnx_Dml


### 2023-06-18
- Novos modelos v2 pré-treinados: 32k e 48k
- Correção de erros de inferência de modelo não-f0
- Para conjuntos de treinamento que excedam 1 hora, faça minibatch-kmeans automáticos para reduzir a forma dos recursos, de modo que o treinamento, a adição e a pesquisa do Index sejam muito mais rápidos.
- Fornecer um espaço de brinquedo vocal2guitar huggingface
- Exclusão automática de áudios de conjunto de treinamento de atalhos discrepantes
- Guia de exportação Onnx

Experimentos com falha:
- ~~Recuperação de recurso: adicionar recuperação de recurso temporal: não eficaz~~
- ~~Recuperação de recursos: adicionar redução de dimensionalidade PCAR: a busca é ainda mais lenta~~
- ~~Aumento de dados aleatórios durante o treinamento: não é eficaz~~

Lista de tarefas：
- ~~Vocos-RVC (vocoder minúsculo): não é eficaz~~
- ~~Suporte de crepe para treinamento: substituído pelo RMVPE~~
- ~~Inferência de crepe de meia precisão：substituída pelo RMVPE. E difícil de conseguir.~~
- Suporte ao editor de F0

### 2023-05-28
- Adicionar notebook jupyter v2, changelog em coreano, corrigir alguns requisitos de ambiente
- Adicionar consoante sem voz e modo de proteção de respiração
- Suporte à detecção de pitch crepe-full
- Separação vocal UVR5: suporte a modelos dereverb e modelos de-echo
- Adicionar nome e versão do experimento no nome do Index
- Suporte aos usuários para selecionar manualmente o formato de exportação dos áudios de saída durante o processamento de conversão de voz em lote e a separação vocal UVR5
- Não há mais suporte para o treinamento do modelo v1 32k

### 2023-05-13
- Limpar os códigos redundantes na versão antiga do tempo de execução no pacote de um clique: lib.infer_pack e uvr5_pack
- Correção do bug de pseudo multiprocessamento no pré-processamento do conjunto de treinamento
- Adição do ajuste do raio de filtragem mediana para o algoritmo de reconhecimento de inclinação da extração
- Suporte à reamostragem de pós-processamento para exportação de áudio
- A configuração "n_cpu" de multiprocessamento para treinamento foi alterada de "extração de f0" para "pré-processamento de dados e extração de f0"
- Detectar automaticamente os caminhos de Index na pasta de registros e fornecer uma função de lista suspensa
- Adicionar "Perguntas e respostas frequentes" na página da guia (você também pode consultar o wiki do RVC no github)
- Durante a inferência, o pitch da colheita é armazenado em cache quando se usa o mesmo caminho de áudio de entrada (finalidade: usando a extração do pitch da colheita, todo o pipeline passará por um processo longo e repetitivo de extração do pitch. Se o armazenamento em cache não for usado, os usuários que experimentarem diferentes configurações de raio de filtragem de timbre, Index e mediana de pitch terão um processo de espera muito doloroso após a primeira inferência)

### 2023-05-14
- Use o envelope de volume da entrada para misturar ou substituir o envelope de volume da saída (pode aliviar o problema de "muting de entrada e ruído de pequena amplitude de saída"). Se o ruído de fundo do áudio de entrada for alto, não é recomendável ativá-lo, e ele não é ativado por padrão (1 pode ser considerado como não ativado)
- Suporte ao salvamento de modelos pequenos extraídos em uma frequência especificada (se você quiser ver o desempenho em épocas diferentes, mas não quiser salvar todos os pontos de verificação grandes e extrair manualmente modelos pequenos pelo processamento ckpt todas as vezes, esse recurso será muito prático)
- Resolver o problema de "erros de conexão" causados pelo proxy global do servidor, definindo variáveis de ambiente
- Oferece suporte a modelos v2 pré-treinados (atualmente, apenas as versões 40k estão disponíveis publicamente para teste e as outras duas taxas de amostragem ainda não foram totalmente treinadas)
- Limita o volume excessivo que excede 1 antes da inferência
- Ajustou ligeiramente as configurações do pré-processamento do conjunto de treinamento


#######################

Histórico de registros de alterações:

### 2023-04-09
- Parâmetros de treinamento corrigidos para melhorar a taxa de utilização da GPU: A100 aumentou de 25% para cerca de 90%, V100: 50% para cerca de 90%, 2060S: 60% para cerca de 85%, P40: 25% para cerca de 95%; melhorou significativamente a velocidade de treinamento
- Parâmetro alterado: total batch_size agora é por GPU batch_size
- Total_epoch alterado: limite máximo aumentado de 100 para 1000; padrão aumentado de 10 para 20
- Corrigido o problema da extração de ckpt que reconhecia o pitch incorretamente, causando inferência anormal
- Corrigido o problema do treinamento distribuído que salvava o ckpt para cada classificação
- Aplicada a filtragem de recursos nan para extração de recursos
- Corrigido o problema com a entrada/saída silenciosa que produzia consoantes aleatórias ou ruído (os modelos antigos precisavam ser treinados novamente com um novo conjunto de dados)

### Atualização 2023-04-16
- Adicionada uma mini-GUI de alteração de voz local em tempo real, iniciada com um clique duplo em go-realtime-gui.bat
- Filtragem aplicada para bandas de frequência abaixo de 50 Hz durante o treinamento e a inferência
- Diminuição da extração mínima de tom do pyworld do padrão 80 para 50 para treinamento e inferência, permitindo que vozes masculinas de tom baixo entre 50-80 Hz não sejam silenciadas
- A WebUI suporta a alteração de idiomas de acordo com a localidade do sistema (atualmente suporta en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW; o padrão é en_US se não for suportado)
- Correção do reconhecimento de algumas GPUs (por exemplo, falha no reconhecimento da V100-16G, falha no reconhecimento da P4)

### Atualização de 2023-04-28
- Atualizadas as configurações do Index faiss para maior velocidade e qualidade
- Removida a dependência do total_npy; o futuro compartilhamento de modelos não exigirá a entrada do total_npy
- Restrições desbloqueadas para as GPUs da série 16, fornecendo configurações de inferência de 4 GB para GPUs com VRAM de 4 GB
- Corrigido o erro na separação do acompanhamento vocal do UVR5 para determinados formatos de áudio
- A mini-GUI de alteração de voz em tempo real agora suporta modelos de pitch não 40k e que não são lentos

### Planos futuros:
Recursos:
- Opção de adição: extrair modelos pequenos para cada epoch salvo
- Adicionar opção: exportar mp3 adicional para o caminho especificado durante a inferência
- Suporte à guia de treinamento para várias pessoas (até 4 pessoas)

Modelo básico:
- Coletar arquivos wav de respiração para adicionar ao conjunto de dados de treinamento para corrigir o problema de sons de respiração distorcidos
- No momento, estamos treinando um modelo básico com um conjunto de dados de canto estendido, que será lançado no futuro