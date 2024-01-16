Instruções e dicas para treinamento RVC
======================================
Estas DICAS explicam como o treinamento de dados é feito.

# Fluxo de treinamento
Explicarei ao longo das etapas na guia de treinamento da GUI.

## Passo 1
Defina o nome do experimento aqui.

Você também pode definir aqui se o modelo deve levar em consideração o pitch.
Se o modelo não considerar o tom, o modelo será mais leve, mas não será adequado para cantar.

Os dados de cada experimento são colocados em `/logs/nome-do-seu-modelo/`.

## Passo 2a
Carrega e pré-processa áudio.

### Carregar áudio
Se você especificar uma pasta com áudio, os arquivos de áudio dessa pasta serão lidos automaticamente.
Por exemplo, se você especificar `C:Users\hoge\voices`, `C:Users\hoge\voices\voice.mp3` será carregado, mas `C:Users\hoge\voices\dir\voice.mp3` será Não carregado.

Como o ffmpeg é usado internamente para leitura de áudio, se a extensão for suportada pelo ffmpeg, ela será lida automaticamente.
Após converter para int16 com ffmpeg, converta para float32 e normalize entre -1 e 1.

### Eliminar ruído
O áudio é suavizado pelo filtfilt do scipy.

### Divisão de áudio
Primeiro, o áudio de entrada é dividido pela detecção de partes de silêncio que duram mais que um determinado período (max_sil_kept=5 segundos?). Após dividir o áudio no silêncio, divida o áudio a cada 4 segundos com uma sobreposição de 0,3 segundos. Para áudio separado em 4 segundos, após normalizar o volume, converta o arquivo wav para `/logs/nome-do-seu-modelo/0_gt_wavs` e, em seguida, converta-o para taxa de amostragem de 16k para `/logs/nome-do-seu-modelo/1_16k_wavs ` como um arquivo wav.

## Passo 2b
### Extrair pitch
Extraia informações de pitch de arquivos wav. Extraia as informações de pitch (=f0) usando o método incorporado em Parselmouth ou pyworld e salve-as em `/logs/nome-do-seu-modelo/2a_f0`. Em seguida, converta logaritmicamente as informações de pitch para um número inteiro entre 1 e 255 e salve-as em `/logs/nome-do-seu-modelo/2b-f0nsf`.

### Extrair feature_print
Converta o arquivo wav para incorporação antecipadamente usando HuBERT. Leia o arquivo wav salvo em `/logs/nome-do-seu-modelo/1_16k_wavs`, converta o arquivo wav em recursos de 256 dimensões com HuBERT e salve no formato npy em `/logs/nome-do-seu-modelo/3_feature256`.

## Passo 3
treinar o modelo.
### Glossário para iniciantes
No aprendizado profundo, o conjunto de dados é dividido e o aprendizado avança aos poucos. Em uma atualização do modelo (etapa), os dados batch_size são recuperados e previsões e correções de erros são realizadas. Fazer isso uma vez para um conjunto de dados conta como um epoch.

Portanto, o tempo de aprendizagem é o tempo de aprendizagem por etapa x (o número de dados no conjunto de dados/tamanho do lote) x o número de epoch. Em geral, quanto maior o tamanho do lote, mais estável se torna o aprendizado (tempo de aprendizado por etapa ÷ tamanho do lote) fica menor, mas usa mais memória GPU. A RAM da GPU pode ser verificada com o comando nvidia-smi. O aprendizado pode ser feito em pouco tempo aumentando o tamanho do lote tanto quanto possível de acordo com a máquina do ambiente de execução.

### Especifique o modelo pré-treinado
O RVC começa a treinar o modelo a partir de pesos pré-treinados em vez de 0, para que possa ser treinado com um pequeno conjunto de dados.

Por padrão

- Se você considerar o pitch, ele carrega `rvc-location/pretrained/f0G40k.pth` e `rvc-location/pretrained/f0D40k.pth`.
- Se você não considerar o pitch, ele carrega `rvc-location/pretrained/f0G40k.pth` e `rvc-location/pretrained/f0D40k.pth`.

Ao aprender, os parâmetros do modelo são salvos em `logs/nome-do-seu-modelo/G_{}.pth` e `logs/nome-do-seu-modelo/D_{}.pth` para cada save_every_epoch, mas especificando nesse caminho, você pode começar a aprender. Você pode reiniciar ou iniciar o treinamento a partir de weights de modelo aprendidos em um experimento diferente.

### Index de aprendizado
O RVC salva os valores de recursos do HuBERT usados durante o treinamento e, durante a inferência, procura valores de recursos que sejam semelhantes aos valores de recursos usados durante o aprendizado para realizar a inferência. Para realizar esta busca em alta velocidade, o index é aprendido previamente.
Para aprendizagem de index, usamos a biblioteca de pesquisa de associação de áreas aproximadas faiss. Leia o valor do recurso `logs/nome-do-seu-modelo/3_feature256` e use-o para aprender o index, e salve-o como `logs/nome-do-seu-modelo/add_XXX.index`.

(A partir da versão 20230428update, ele é lido do index e não é mais necessário salvar/especificar.)

### Descrição do botão
- Treinar modelo: Após executar o passo 2b, pressione este botão para treinar o modelo.
- Treinar índice de recursos: após treinar o modelo, execute o aprendizado do index.
- Treinamento com um clique: etapa 2b, treinamento de modelo e treinamento de index de recursos, tudo de uma vez.