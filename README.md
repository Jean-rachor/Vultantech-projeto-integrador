Este guia mostra como iniciar a aplicação no Windows, abrir a página local e encerrar o serviço.

## 1. Abrir o projeto

Abra o PowerShell e entre na pasta do projeto:

```powershell
cd C:\caminho\para\Vulcantech
```

É necessário ter Python 3.10 ou superior instalado. Se a pasta `.venv` ainda não existir, crie o ambiente virtual:

```powershell
py -3 -m venv .venv
```

## 2. Instalar as dependências

Execute este comando na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Na primeira execução, prepare o banco SQLite e cadastre as imagens existentes (a API também inicializa um catálogo vazio ao iniciar):

```powershell
.\.venv\Scripts\python.exe Vulcantech_imagens_soldas\criar_db.py
```

Esse comando pode ser repetido. Ele não duplica as imagens já cadastradas.

## 3. Iniciar a aplicação

Na mesma janela do PowerShell, rode:

```powershell
.\.venv\Scripts\uvicorn.exe main:app --app-dir src --host 127.0.0.1 --port 8000
```

Deixe essa janela aberta enquanto usar a aplicação. Abra o navegador e acesse:

**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

## 4. Usar a página

1. Confira no cabeçalho se a API está disponível.
2. O **Centro de operações · SCADA** mostra uma reprodução somente leitura do CSV histórico em `Data/dados_sensores_vulcantech.csv`. As condições exibidas são derivadas do campo `tipo_falha`; elas não são alarmes ao vivo nem conexão com CLP/PLC.
3. Em **Inspeção visual**, selecione uma imagem JPEG, PNG ou WebP e escolha **Imagem óptica comum**.
4. Clique em **Analisar imagem**. A API valida o arquivo, processa-o com OpenCV, compara descritores ORB com as referências originais e armazena o upload original no SQLite. Métricas experimentais, referências e limitações aparecem na página e a inspeção entra no histórico.
5. Se houver confirmação de inspetor qualificado, registre a classe e a origem do rótulo no resultado para começar a montar uma base visual supervisionada.
6. Para ver o estado do serviço, abra `/health`. A lista de features do modelo aparece em `/model-info` quando o artefato estiver disponível. O estado e os dados necessários para um classificador visual estão em `/vision-model-info`.
7. A seção **Predição de sensores** requer valores válidos para todas as features listadas por `/model-info`. Use **Preencher exemplo** para carregar um JSON base ou baixe [`predicao_sensores_base.json`](Docs/Exemplos/predicao_sensores_base.json). O exemplo usa uma linha do CSV demonstrativo do projeto e serve para conferir o formato; substitua pelos dados medidos da máquina antes de interpretar uma previsão. O endpoint também aceita lotes; a documentação de requisição está em `/docs`.
8. Em **Histórico de inspeções**, clique em **Atualizar histórico** para consultar análises registradas.

## 5. Endereços úteis

- Interface: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Documentação interativa da API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Estado da API e do modelo: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Features necessárias para predição: [http://127.0.0.1:8000/model-info](http://127.0.0.1:8000/model-info)
- Disponibilidade das referências visuais: [http://127.0.0.1:8000/references](http://127.0.0.1:8000/references)
- Histórico em JSON: [http://127.0.0.1:8000/inspections](http://127.0.0.1:8000/inspections)
- JSON de exemplo da predição: [http://127.0.0.1:8000/examples/sensor-prediction.json](http://127.0.0.1:8000/examples/sensor-prediction.json)
- Resumo de células SCADA simulado: [http://127.0.0.1:8000/scada/overview](http://127.0.0.1:8000/scada/overview)
- Série histórica SCADA simulada: [http://127.0.0.1:8000/scada/telemetry?cell=CEL-202](http://127.0.0.1:8000/scada/telemetry?cell=CEL-202)

## 6. Encerrar

Volte à janela do PowerShell onde o servidor está rodando e pressione **Ctrl+C**.

## Observações

- A API fica acessível somente no próprio computador enquanto o servidor estiver em execução.
- O modelo disponível classifica dados de sensores, não imagens. A inspeção visual atual mede contornos e compara referências, mas não classifica defeitos. Para treinar esse classificador, são necessárias imagens com rótulos confirmados por classe; os uploads ficam sem rótulo por padrão.
- Uploads válidos são armazenados no banco SQLite como BLOBs. O tamanho do banco cresce com novas imagens; proteja o arquivo e defina retenção e descarte para o uso pretendido.
- Imagens sem rótulo confiável são registradas como `NAO_CLASSIFICADA`.
- O catálogo atual tem 20 imagens originais de referência, sem rótulos confiáveis. A comparação ORB é exploratória e não classifica defeitos.
- A segmentação HSV em fotos comuns é experimental; ela não mede temperatura. A termografia real fica indisponível sem imagem calibrada e parâmetros de captura.
- A decisão industrial continua `PENDENTE_APROVACAO`. A aplicação não envia comandos a máquinas.
- Se a porta 8000 estiver ocupada, inicie com `--port 8001` e acesse `http://127.0.0.1:8001/`.
- Se a tela mostrar “Não foi possível conectar à API local”, inicie o servidor no PowerShell, mantenha a janela aberta e entre pelo endereço `http://127.0.0.1:8000/`. Não abra o `index.html` diretamente pelo Explorador de Arquivos. Após atualizar o código, encerre o servidor com Ctrl+C e inicie-o novamente.

Para detalhes de instalação, endpoints, imagens e Docker, consulte [`Docs/Entregaveis/Documentos/Operacao_API.md`](Docs/Entregaveis/Documentos/Operacao_API.md).
