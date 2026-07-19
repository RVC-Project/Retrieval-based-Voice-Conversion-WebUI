# <b>FAQ AI HUB BRASIL</b>
## <span style="color: #337dff;">O que é epoch, quantos utilizar, quanto de dataset utilizar e qual à configuração interessante?</span>
Epochs basicamente quantas vezes o seu dataset foi treinado.

Recomendado ler Q8 e Q9 no final dessa página pra entender mais sobre dataset e epochs

__**Não é uma regra, mas opinião:**__

### **Mangio-Crepe Hop Length**
- 64 pra cantores e dubladores
- 128(padrão) para os demais (editado)

### **Epochs e dataset**
600epoch para cantores - --dataset entre 10 e 50 min  desnecessario mais que 50 minutos--
300epoch para os demais - --dataset entre 10 e 50 min desnecessario mais que 50 minutos--

### **Tom**
magio-crepe se for audios extraído de alguma musica
harvest se for de estúdio<hr>

## <span style="color: #337dff;">O que é index?</span>
Basicamente o que define o sotaque. Quanto maior o numero, mas próximo o sotaque fica do original. Porém, quando o modelo é bem, não é necessário um index.<hr>

## <span style="color: #337dff;">O que significa cada sigla (pm, harvest, crepe, magio-crepe, RMVPE)?</span>

- pm = extração mais rápida, mas discurso de qualidade inferior;
- harvest = graves melhores, mas extremamente lentos;
- dio = conversão rápida mas pitch ruim;
- crepe = melhor qualidade, mas intensivo em GPU;
- crepe-tiny = mesma coisa que o crepe, só que com a qualidade um pouco inferior;
- **mangio-crepe = melhor qualidade, mais otimizado; (MELHOR OPÇÃO)**
- mangio-crepe-tiny = mesma coisa que o mangio-crepe, só que com a qualidade um pouco inferior;
- RMVPE: um modelo robusto para estimativa de afinação vocal em música polifônica;<hr>

## <span style="color: #337dff;">Pra rodar localmente, quais os requisitos minimos?</span>
Já tivemos relatos de pessoas com GTX 1050 rodando inferencia, se for treinar numa 1050 vai demorar muito mesmo e inferior a isso, normalmente da tela azul

O mais importante é placa de vídeo, vram na verdade
Se você tiver 4GB ou mais, você tem uma chance.

**NOS DOIS CASOS NÃO É RECOMENDADO UTILIZAR O PC ENQUANTO ESTÁ UTILIZNDO, CHANCE DE TELA AZUL É ALTA**
### Inference
Não é algo oficial para requisitos minimos
- Placa de vídeo: nvidia de 4gb
- Memoria ram: 8gb
- CPU: ?
- Armanezamento: 20gb (sem modelos)

### Treinamento de voz
Não é algo oficial para requisitos minimos
- Placa de vídeo: nvidia de 6gb
- Memoria ram: 16gb
- CPU: ?
- Armanezamento: 20gb (sem modelos)<hr>

## <span style="color: #337dff;">Limite de GPU no Google Colab excedido, apenas CPU o que fazer?</span>
Recomendamos esperar outro dia pra liberar mais 15gb ou 12 horas pra você. Ou você pode contribuir com o Google pagando algum dos planos, ai aumenta seu limite.<br>
Utilizar apenas CPU no Google Colab demora DEMAIS.<hr>


## <span style="color: #337dff;">Google Colab desconectando com muita frequencia, o que fazer?</span>
Neste caso realmente não tem muito o que fazer. Apenas aguardar o proprietário do código corrigir ou a gente do AI HUB Brasil achar alguma solução. Isso acontece por diversos motivos, um incluindo a Google barrando o treinamento de voz.<hr>

## <span style="color: #337dff;">O que é Batch Size/Tamanho de lote e qual numero utilizar?</span>
Batch Size/Tamanho do lote é basicamente quantos epoch faz ao mesmo tempo. Se por 20, ele fazer 20 epoch ao mesmo tempo e isso faz pesar mais na máquina e etc.<br>

No Google Colab você pode utilizar até 20 de boa.<br>
Se rodando localmente, depende da sua placa de vídeo, começa por baixo (6) e vai testando.<hr>

## <span style="color: #337dff;">Sobre backup na hora do treinamento</span>
Backup vai de cada um. Eu quando uso a ``easierGUI`` utilizo a cada 100 epoch (meu caso isolado).
No colab, se instavel, coloque a cada 10 epoch
Recomendo utilizarem entre 25 e 50 pra garantir.

Lembrando que cada arquivo geral é por volta de 50mb, então tenha muito cuidado quanto você coloca. Pois assim pode acabar lotando seu Google Drive ou seu PC.

Depois de finalizado, da pra apagar os epoch de backup.<hr>

## <span style="color: #337dff;">Como continuar da onde parou pra fazer mais epochs?</span>
Primeira coisa que gostaria de lembrar, não necessariamente quanto mais epochs melhor. Se fizer epochs demais vai dar **overtraining** o que pode ser ruim.

### GUI NORMAL
- Inicie normalmente a GUI novamente.
- Na aba de treino utilize o MESMO nome que estava treinando, assim vai continuar o treino onde parou o ultimo backup.
- Ignore as opções ``Processar o Conjunto de dados`` e ``Extrair Tom``
- Antes de clicar pra treinar, arrume os epoch, bakcup e afins. 
    - Obviamente tem que ser um numero maior do qu estava em epoch.
    - Backup você pode aumentar ou diminuir
- Agora você vai ver a opção ``Carregue o caminho G do modelo base pré-treinado:`` e ``Carregue o caminho D do modelo base pré-treinado:``
    -Aqui você vai por o caminho dos modelos que estão em ``./logs/minha-voz``
        - Vai ficar algo parecido com isso ``e:/RVC/logs/minha-voz/G_0000.pth`` e ``e:/RVC/logs/minha-voz/D_0000.pth``
-Coloque pra treinar

**Lembrando que a pasta logs tem que ter todos os arquivos e não somente o arquivo ``G`` e ``D``**

### EasierGUI
- Inicie normalmente a easierGUI novamente.
- Na aba de treino utilize o MESMO nome que estava treinando, assim vai continuar o treino onde parou o ultimo backup.
- Selecione 'Treinar modelo', pode pular os 2 primeiros passos já que vamos continuar o treino.<hr><br>


# <b>FAQ Original traduzido</b>
## <b><span style="color: #337dff;">Q1: erro ffmpeg/erro utf8.</span></b>
Provavelmente não é um problema do FFmpeg, mas sim um problema de caminho de áudio;

O FFmpeg pode encontrar um erro ao ler caminhos contendo caracteres especiais como spaces e (), o que pode causar um erro FFmpeg; e quando o áudio do conjunto de treinamento contém caminhos chineses, gravá-lo em filelist.txt pode causar um erro utf8.<hr>

## <b><span style="color: #337dff;">Q2:Não é possível encontrar o arquivo de Index após "Treinamento com um clique".</span></b>
Se exibir "O treinamento está concluído. O programa é fechado ", então o modelo foi treinado com sucesso e os erros subsequentes são falsos;

A falta de um arquivo de index 'adicionado' após o treinamento com um clique pode ser devido ao conjunto de treinamento ser muito grande, fazendo com que a adição do index fique presa; isso foi resolvido usando o processamento em lote para adicionar o index, o que resolve o problema de sobrecarga de memória ao adicionar o index. Como solução temporária, tente clicar no botão "Treinar Index" novamente.<hr>

## <b><span style="color: #337dff;">Q3:Não é possível encontrar o modelo em “Modelo de voz” após o treinamento</span></b>
Clique em "Atualizar lista de voz" ou "Atualizar na EasyGUI e verifique novamente; se ainda não estiver visível, verifique se há erros durante o treinamento e envie capturas de tela do console, da interface do usuário da Web e dos ``logs/experiment_name/*.log`` para os desenvolvedores para análise posterior.<hr>

## <b><span style="color: #337dff;">Q4:Como compartilhar um modelo/Como usar os modelos dos outros?</span></b>
Os arquivos ``.pth`` armazenados em ``*/logs/minha-voz`` não são destinados para compartilhamento ou inference, mas para armazenar os checkpoits do experimento para reprodutibilidade e treinamento adicional. O modelo a ser compartilhado deve ser o arquivo ``.pth`` de 60+MB na pasta **weights**;

No futuro, ``weights/minha-voz.pth`` e ``logs/minha-voz/added_xxx.index`` serão mesclados em um único arquivo de ``weights/minha-voz.zip`` para eliminar a necessidade de entrada manual de index; portanto, compartilhe o arquivo zip, não somente o arquivo .pth, a menos que você queira continuar treinando em uma máquina diferente;

Copiar/compartilhar os vários arquivos .pth de centenas de MB da pasta de logs para a pasta de weights para inference forçada pode resultar em erros como falta de f0, tgt_sr ou outras chaves. Você precisa usar a guia ckpt na parte inferior para manualmente ou automaticamente (se as informações forem encontradas nos ``logs/minha-voz``), selecione se deseja incluir informações de tom e opções de taxa de amostragem de áudio de destino e, em seguida, extrair o modelo menor. Após a extração, haverá um arquivo pth de 60+ MB na pasta de weights, e você pode atualizar as vozes para usá-lo.<hr>

## <b><span style="color: #337dff;">Q5 Erro de conexão:</span></b>
Para sermos otimistas, aperte F5/recarregue a página, pode ter sido apenas um bug da GUI

Se não...
Você pode ter fechado o console (janela de linha de comando preta).
Ou o Google Colab, no caso do Colab, as vezes pode simplesmente fechar<hr>

## <b><span style="color: #337dff;">Q6: Pop-up WebUI 'Valor esperado: linha 1 coluna 1 (caractere 0)'.</span></b>
Desative o proxy LAN do sistema/proxy global e atualize.<hr>

## <b><span style="color: #337dff;">Q7:Como treinar e inferir sem a WebUI?</span></b>
Script de treinamento:
<br>Você pode executar o treinamento em WebUI primeiro, e as versões de linha de comando do pré-processamento e treinamento do conjunto de dados serão exibidas na janela de mensagens.<br>

Script de inference:
<br>https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/myinfer.py<br>


por exemplo<br>

``runtime\python.exe myinfer.py 0 "E:\audios\1111.wav" "E:\RVC\logs\minha-voz\added_IVF677_Flat_nprobe_7.index" harvest "test.wav" "weights/mi-test.pth" 0.6 cuda:0 True``<br>


f0up_key=sys.argv[1]<br>
input_path=sys.argv[2]<br>
index_path=sys.argv[3]<br>
f0method=sys.argv[4]#harvest or pm<br>
opt_path=sys.argv[5]<br>
model_path=sys.argv[6]<br>
index_rate=float(sys.argv[7])<br>
device=sys.argv[8]<br>
is_half=bool(sys.argv[9])<hr>

## <b><span style="color: #337dff;">Q8: Erro Cuda/Cuda sem memória.</span></b>
Há uma pequena chance de que haja um problema com a configuração do CUDA ou o dispositivo não seja suportado; mais provavelmente, não há memória suficiente (falta de memória).<br>

Para treinamento, reduza o (batch size) tamanho do lote (se reduzir para 1 ainda não for suficiente, talvez seja necessário alterar a placa gráfica); para inference, ajuste as configurações x_pad, x_query, x_center e x_max no arquivo config.py conforme necessário. Cartões de memória 4G ou inferiores (por exemplo, 1060(3G) e várias placas 2G) podem ser abandonados, enquanto os placas de vídeo com memória 4G ainda têm uma chance.<hr>

## <b><span style="color: #337dff;">Q9:Quantos total_epoch são ótimos?</span></b>
Se a qualidade de áudio do conjunto de dados de treinamento for ruim e o nível de ruído for alto, **20-30 epochs** são suficientes. Defini-lo muito alto não melhorará a qualidade de áudio do seu conjunto de treinamento de baixa qualidade.<br>

Se a qualidade de áudio do conjunto de treinamento for alta, o nível de ruído for baixo e houver duração suficiente, você poderá aumentá-lo. **200 é aceitável** (uma vez que o treinamento é rápido e, se você puder preparar um conjunto de treinamento de alta qualidade, sua GPU provavelmente poderá lidar com uma duração de treinamento mais longa sem problemas).<hr>

## <b><span style="color: #337dff;">Q10:Quanto tempo de treinamento é necessário?</span></b>

**Recomenda-se um conjunto de dados de cerca de 10 min a 50 min.**<br>

Com garantia de alta qualidade de som e baixo ruído de fundo, mais pode ser adicionado se o timbre do conjunto de dados for uniforme.<br>

Para um conjunto de treinamento de alto nível (limpo + distintivo), 5min a 10min é bom.<br>

Há algumas pessoas que treinaram com sucesso com dados de 1 a 2 minutos, mas o sucesso não é reproduzível por outros e não é muito informativo. <br>Isso requer que o conjunto de treinamento tenha um timbre muito distinto (por exemplo, um som de menina de anime arejado de alta frequência) e a qualidade do áudio seja alta;
Dados com menos de 1 minuto, já obtivemo sucesso. Mas não é recomendado.<hr>


## <b><span style="color: #337dff;">Q11:Qual é a taxa do index e como ajustá-la?</span></b>
Se a qualidade do tom do modelo pré-treinado e da fonte de inference for maior do que a do conjunto de treinamento, eles podem trazer a qualidade do tom do resultado do inference, mas ao custo de um possível viés de tom em direção ao tom do modelo subjacente/fonte de inference, em vez do tom do conjunto de treinamento, que é geralmente referido como "vazamento de tom".<br>

A taxa de index é usada para reduzir/resolver o problema de vazamento de timbre. Se a taxa do index for definida como 1, teoricamente não há vazamento de timbre da fonte de inference e a qualidade do timbre é mais tendenciosa em relação ao conjunto de treinamento. Se o conjunto de treinamento tiver uma qualidade de som mais baixa do que a fonte de inference, uma taxa de index mais alta poderá reduzir a qualidade do som. Reduzi-lo a 0 não tem o efeito de usar a mistura de recuperação para proteger os tons definidos de treinamento.<br>

Se o conjunto de treinamento tiver boa qualidade de áudio e longa duração, aumente o total_epoch, quando o modelo em si é menos propenso a se referir à fonte inferida e ao modelo subjacente pré-treinado, e há pouco "vazamento de tom", o index_rate não é importante e você pode até não criar/compartilhar o arquivo de index.<hr>

## <b><span style="color: #337dff;">Q12:Como escolher o GPU ao inferir?</span></b>
No arquivo ``config.py``, selecione o número da placa em "device cuda:".<br>

O mapeamento entre o número da placa e a placa gráfica pode ser visto na seção de informações da placa gráfica da guia de treinamento.<hr>

## <b><span style="color: #337dff;">Q13:Como usar o modelo salvo no meio do treinamento?</span></b>
Salvar via extração de modelo na parte inferior da guia de processamento do ckpt.<hr>

## <b><span style="color: #337dff;">Q14: Erro de arquivo/memória (durante o treinamento)?</span></b>
Muitos processos e sua memória não é suficiente. Você pode corrigi-lo por:

1. Diminuir a entrada no campo "Threads da CPU".
2. Diminuir o tamanho do conjunto de dados.

## Q15: Como continuar treinando usando mais dados

passo 1: coloque todos os dados wav no path2.

etapa 2: exp_name2 + path2 -> processar conjunto de dados e extrair recurso.

passo 3: copie o arquivo G e D mais recente de exp_name1 (seu experimento anterior) para a pasta exp_name2.

passo 4: clique em "treinar o modelo" e ele continuará treinando desde o início da época anterior do modelo exp.

## Q16: erro sobre llvmlite.dll

OSError: Não foi possível carregar o arquivo de objeto compartilhado: llvmlite.dll

FileNotFoundError: Não foi possível encontrar o módulo lib\site-packages\llvmlite\binding\llvmlite.dll (ou uma de suas dependências). Tente usar o caminho completo com sintaxe de construtor.

O problema acontecerá no Windows, instale https://aka.ms/vs/17/release/vc_redist.x64.exe e será corrigido.

## Q17: RuntimeError: O tamanho expandido do tensor (17280) deve corresponder ao tamanho existente (0) na dimensão 1 não singleton. Tamanhos de destino: [1, 17280]. Tamanhos de tensor: [0]

Exclua os arquivos wav cujo tamanho seja significativamente menor que outros e isso não acontecerá novamente. Em seguida, clique em "treinar o modelo" e "treinar o índice".

## Q18: RuntimeError: O tamanho do tensor a (24) deve corresponder ao tamanho do tensor b (16) na dimensão não singleton 2

Não altere a taxa de amostragem e continue o treinamento. Caso seja necessário alterar, o nome do exp deverá ser alterado e o modelo será treinado do zero. Você também pode copiar o pitch e os recursos (pastas 0/1/2/2b) extraídos da última vez para acelerar o processo de treinamento.

