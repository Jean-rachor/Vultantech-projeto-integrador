# VulcanTech — API e interface local

## Iniciar

Na raiz do projeto, use Python 3.10 ou superior:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Vulcantech_imagens_soldas\criar_db.py
.\.venv\Scripts\uvicorn.exe main:app --app-dir src --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000/`. A documentação OpenAPI fica em `/docs` e `/openapi.json`. Para instalar as dependências de teste, use `requirements-test.txt`; para scripts de treinamento e gráficos, instale também `requirements-training.txt`.

## Banco de imagens e inspeções

`Vulcantech_imagens_soldas/criar_db.py` cria `vulcantech.sqlite3` na raiz e registra os arquivos locais, derivados e externos como BLOBs com SHA-256, caminho relativo, origem, metadados disponíveis e rótulo. Imagens sem proveniência de rótulo confiável recebem `NAO_CLASSIFICADA`. Derivadas herdam a referência à imagem de origem, não uma classe. Depois de uma inspeção válida, a API também guarda o upload original como BLOB em `image_assets`, com hash, nome seguro, metadados de imagem e estado `aguardando_rotulo_confiavel`. O arquivo não é gravado em uma pasta pública. Uploads repetidos com o mesmo SHA-256 reutilizam o registro da imagem.

`Vulcantech_imagens_soldas/gerar_variacoes.py` cria duas mudanças leves de luminosidade por imagem original, sem alteração morfológica, e registra a transformação, hashes e rótulo não classificado em `derivadas/manifest.json`. Faixas brancas contínuas periféricas são removidas somente se detectadas; as 20 imagens atuais são praticamente sem borda branca, portanto não foram recortadas.

Imagens externas em `Vulcantech_imagens_soldas/externas` são apenas referências visuais. A procedência, autoria, licença e descrição da fonte estão em `manifest.json`. Não fazem parte do treino ou validação e permanecem sem classe confirmada.

## Endpoints

- `GET /health`: estado da API, do artefato preditivo, do SQLite e das regras pendentes.
- `GET /references`: quantidade de referências originais legíveis, quantidade de rótulos com fonte e disponibilidade para comparação; não expõe BLOBs.
- `GET /vision-model-info`: estado do classificador visual e requisito de dados rotulados. Hoje não há classificador visual treinado.
- `PUT /images/{id}/label`: registra rótulo confirmado por pessoa em imagem original ou upload. Aceita `NORMAL`, `POROSIDADE`, `TRINCA` ou `DESCONTINUIDADE` e exige `label_source` com a procedência. O registro apoia a formação de um conjunto de treino, mas não treina modelo automaticamente.
- `GET /model-info`: features e seleção do modelo quando o artefato puder ser carregado.
- `GET /examples/sensor-prediction.json`: baixa o corpo-base de uma predição. Os números são demonstrativos, derivados do CSV do projeto; substitua-os por medições reais antes de interpretar uma previsão.
- `POST /inspect`: recebe `file` multipart e `input_type=optical|thermal`. Valida o MIME contra o formato real, integridade, limite configurável de bytes (12 MB por padrão) e até 20 megapixels; decodifica a imagem com OpenCV. A entrada óptica retorna métricas Canny/contornos, segmentação HSV/Otsu experimental e comparação ORB exploratória com referências do SQLite. A entrada térmica é validada, mas retorna indisponível sem calibração e escala térmica.
- `POST /predict`: recebe `{"features": { ... }}` com as features exigidas por `/model-info`.
- `POST /predict-batch`: recebe `{"rows": [{ ... }, ...]}`; máximo de 500 linhas por chamada. Um registro inválido retorna decisão pendente no respectivo item, sem descartar os demais resultados.
- `GET /inspections?limit=50&offset=0`: consulta análises registradas.

## Comparação com imagens de referência

O banco é SQLite. `image_assets` armazena imagem em `image_blob` (BLOB), SHA-256, caminho de origem, tipo, rótulo, fonte do rótulo e `metadata_json`. `inspections` e `decision_audit` registram resultados e validação operacional. A aplicação inicializa o esquema existente e cadastra as imagens do projeto se o catálogo estiver sem referências originais.

Para comparação são consultadas somente imagens `source_type=projeto`; derivadas são excluídas para não duplicar a mesma origem, e imagens externas são excluídas por condições de captura/processo diferentes. O banco atual contém 20 referências originais e todas estão `NAO_CLASSIFICADA`, sem fonte de rótulo. Seus BLOBs são decodificados com OpenCV, imagens ilegíveis são ignoradas, e os descritores ORB com teste de razão Hamming (0,75) ordenam até cinco correspondências. O catálogo decodificado é mantido em cache durante a execução; reinicie a API após mudanças em `image_assets`.

A comparação é experimental: correspondência local não significa mesma peça, aprovação ou defeito. O sistema não retorna confiança de classificação, e todo resultado visual segue `NAO_CLASSIFICADA` até que existam rótulos confiáveis, condições comparáveis e critérios validados. A interface permite registrar uma classe confirmada por inspetor e sua fonte em imagens originais ou uploads. Isso forma um conjunto de dados para trabalho futuro, mas não altera nem treina o modelo. Uploads válidos ficam em BLOB no SQLite; inspeções guardam hash, métricas, limitações e decisão. O banco cresce com novas imagens, portanto defina retenção e proteja o arquivo e seus backups antes de uso compartilhado.

O modelo selecionado é Random Forest pelo critério já usado no código: maior recall macro no teste temporal. O artefato existente contém os três modelos e o scaler, mas não registra a seleção e métricas em seu próprio metadado. A seleção é documentada no código e na comparação de métricas existente. Uma previsão individual não é uma nova validação. O endpoint rejeita ausência, excesso ou valor não finito nas features.

Entradas ausentes, não numéricas, inválidas ou imagens não processáveis são registradas como decisão pendente, com `PENDENTE_DADOS_INVALIDOS`, sem ação. A resposta HTTP 422 inclui o identificador e o registro da decisão.

## Ações e segurança operacional

Diagnóstico e decisão operacional são registrados separadamente. A versão da regra é `pendente-de-aprovacao`; `action` permanece nula e o estado `PENDENTE_APROVACAO`. Entradas inválidas ou não classificadas também não resultam em ação. Nenhum endpoint comanda CLP, altera velocidade ou executa parada de emergência. A recomendação só poderá ser habilitada após aprovação e registro dos critérios pelo responsável.

## Limitações atuais

- As 20 imagens locais não têm classificação comprovada. Os nomes não são usados para inferir defeitos.
- O ORB não realiza alinhamento de perspectiva nem valida equivalência de peça, câmera, escala ou condição de captura; as correspondências servem apenas para exploração visual.
- HAZ em fotografia comum é somente segmentação experimental de regiões HSV, sem temperatura ou conclusão de sobreaquecimento.
- Termografia calibrada não está disponível até que imagens térmicas e escala, calibração e parâmetros de captura sejam fornecidos.
- A API recebe as features de engenharia prontas para a previsão tabular. Não reconstrói as janelas temporais de uma série por célula a partir de uma única leitura.
- O requisito de recall macro ≥ 0,90 aparece no material de avaliação. O gráfico comparativo existente reporta recalls acima da meta; isso é evidência histórica do artefato apresentado, não validação independente nesta entrega.

## Docker

Primeiro crie e popule o banco local com o comando de `criar_db.py`. Depois, na raiz:

```powershell
docker compose up --build
```

A aplicação estará em `http://127.0.0.1:8000/`; a documentação está em `/docs`. O Compose monta o mesmo `vulcantech.sqlite3` na raiz do projeto e publica a porta somente em loopback.

## Testes

```powershell
.\.venv\Scripts\python.exe -m pytest src\tests -q
```

Os testes cobrem a ausência de regras aprovadas e garantem que diagnóstico ausente, inválido ou ambíguo não escolhe ação industrial.
