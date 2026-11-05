# Parecer técnico preliminar — VulcanTech

**Data da revisão:** 29 de outubro de 2026
**Escopo:** protótipo local de inspeção visual experimental de soldas e predição de falhas operacionais por sensores.
**Natureza:** avaliação técnica preliminar do material e do código disponíveis; não substitui parecer jurídico, análise de segurança de máquina, auditoria de conformidade nem validação industrial.

## Síntese

A aplicação agora separa resultados analíticos de decisões operacionais. O modelo tabular pode emitir uma classe e probabilidades. A API grava a ação como nula e a decisão como `PENDENTE_APROVACAO`; nenhuma rota envia comandos a CLP. A inspeção óptica calcula métricas de Canny/contornos e regiões HSV experimentais, mantendo imagens sem procedência de classe como `NAO_CLASSIFICADA`. Imagens comuns não produzem temperatura. A análise termográfica real permanece indisponível sem imagem térmica e calibração.

O protótipo pode apoiar avaliação técnica local, mas não demonstra prontidão para controle fabril, conformidade legal ou ganhos ambientais medidos. Permanecem ausentes regras operacionais aprovadas, rótulos confiáveis para imagens, dados termográficos calibrados, política de retenção e avaliação independente do modelo.

## Evidências e limites do material

- O CSV tem 10.000 linhas e quatro códigos operacionais. A contagem observada é: normal 8.557; desgaste 470; refrigeração 509; desalinhamento/trinca 464. A origem, o processo de rotulagem e a representatividade do CSV para uma linha real não estão registrados nos arquivos consultados.
- O PDF do projeto define divisão temporal 70/30, SMOTE/ponderação, comparação multiclasse e meta de recall macro ≥ 0,90. A figura comparativa existente reporta recall macro acima dessa meta para os três modelos. Esses valores são resultados já presentes no projeto, não uma repetição independente desta revisão.
- O código seleciona o modelo de maior recall macro. A inferência atual usa Random Forest, conforme a seleção e a figura existentes. O artefato antigo não registra versão das bibliotecas, modelo selecionado ou métricas no próprio arquivo; a próxima execução do pipeline passa a gravar seleção e métricas temporais.
- As features de normalização por liga usam a média calculada sobre o conjunto inteiro antes do corte temporal. A interpolação temporal também pode usar observações posteriores ao ponto interpolado. Isso cria risco de vazamento de informação entre treino e teste e limita a interpretação do recall reportado como desempenho operacional futuro. O rolling do pipeline não mantém estado por inspeção e a API recebe as features já engenheiradas. Reavaliar o modelo com pré-processamento ajustado só no treino e inferência reproduzível em tempo real antes de alegar desempenho de produção.
- As 20 imagens locais não têm rótulos com origem comprovada. Foram mantidas como `NAO_CLASSIFICADA`; 40 derivações alteram levemente a luminosidade e também ficam sem rótulo. Duas imagens externas foram incorporadas somente como referências visuais com licença e autoria no manifesto. Nenhuma dessas imagens foi usada para treinar ou validar classificador.

## LGPD

A LGPD incide sobre tratamento de dados pessoais, inclusive digital. Dados puramente técnicos de equipamento não são automaticamente dados pessoais. Imagens ou registros que identifiquem trabalhadores, turnos ou desempenho individual podem tornar-se dados pessoais, dependendo do contexto e da possibilidade de associação. A aplicação não possui finalidade de avaliação de empregados e não implementa reconhecimento de pessoas.

O controlador, os operadores, a hipótese legal, os titulares potencialmente afetados e os prazos de retenção não foram identificados no projeto. Não é possível concluir conformidade ou escolher uma base legal sem essas informações. Se a implantação vier a processar dados de trabalhadores identificáveis, documentar finalidade e hipótese legal, limitar os campos e o enquadramento da câmera, informar os titulares conforme aplicável, controlar acesso, definir retenção e descarte e avaliar a necessidade de relatório de impacto com o encarregado/jurídico. A LGPD prevê revisão de decisões unicamente automatizadas que afetem interesses de pessoas; o escopo atual trata condição de processo e não toma decisão sobre pessoas. Reavaliar se essa finalidade mudar. [Lei nº 13.709/2018, texto compilado](https://www.gov.br/mj/pt-br/assuntos/sua-protecao/sedigi/Lei13709.pdf) e [guia da ANPD sobre agentes de tratamento](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes/guia-orientativo-para-definicoes-dos-agentes-de-tratamento-de-dados-pessoais-e-do-encarregado).

O protótipo limita upload a 12 MB, processa o arquivo em memória e guarda hash, métricas e decisão, sem persistir o upload. O SQLite contém BLOBs das imagens-fonte da base, incluindo as externas; o arquivo do banco não tem criptografia própria. O serviço de desenvolvimento é local e o Compose publica apenas em `127.0.0.1`. Não há autenticação de usuário nem política automática de retenção. Portanto, restringir ACL do arquivo e backups, proteger o host, não expor a porta em rede e definir retenção e descarte antes de uso compartilhado ou produtivo. Hashes de imagem não devem ser tratados como anonimização garantida.

## EU AI Act

O Regulamento (UE) 2024/1689 tem aplicação geral desde 2 de agosto de 2026, com exceções e fases específicas. A alteração europeia de 2026 ajustou a aplicação das obrigações para sistemas de alto risco: 2 de dezembro de 2027 para a categoria do artigo 6(2)/Anexo III e 2 de agosto de 2028 para a categoria do artigo 6(1)/Anexo I. Confirmar a versão consolidada e a situação jurídica no momento de qualquer comercialização ou entrada em serviço na UE. [Texto consolidado do AI Act no EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng), [Regulamento (UE) 2026/1744](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32026R1744).

O artigo 6(1) exige, entre outras condições, que o sistema seja um produto ou componente de segurança coberto por legislação de harmonização do Anexo I e sujeito a avaliação de conformidade por terceiro. O texto consolidado também exclui sistemas usados somente em aspectos não relacionados à segurança, como otimização ou controle de qualidade, exceto quando uma falha possa colocar saúde ou segurança em risco. A finalidade implementada aqui é inspeção/qualidade e apoio à manutenção, sem comando de máquina; isso sugere, mas não determina juridicamente, que o protótipo não é componente de segurança. Uma integração futura cujo mau funcionamento possa afetar segurança deve ser reavaliada com a legislação de máquinas e avaliação de conformidade aplicável. O Regulamento de Máquinas (UE) 2023/1230 tem aplicação geral a partir de 20 de janeiro de 2027. [Resumo oficial de requisitos de segurança de máquinas](https://eur-lex.europa.eu/EN/legal-content/summary/machinery-safety-requirements.html?fromSummary=24).

O Anexo III inclui usos ligados a emprego e gestão de trabalhadores, entre outros contextos. O protótipo não pontua nem monitora trabalhadores. Se a finalidade mudar para medir desempenho individual, distribuir tarefas ou apoiar decisões de emprego, a análise de alto risco muda. Para qualquer categoria que venha a ser classificada como alto risco, completar a análise de risco, governança e qualidade dos dados, documentação técnica, logs, transparência, supervisão humana, robustez, precisão, segurança e processo de conformidade exigidos na fase aplicável. O registro de inspeções implementado cobre identificação, entradas analíticas, versão/estado da regra e ação, mas não constitui por si só conformidade com o AI Act.

## Cibersegurança OT/IT e segurança industrial

A API não implementa conexão com CLP e não envia comandos. O serviço deve permanecer em rede local de desenvolvimento. Para piloto, posicioná-lo fora da rede de controle, limitar rotas e portas, usar segmentação e uma DMZ/gateway administrada, aplicar allowlist de origem, autenticação e autorização para qualquer uso multiusuário, gestão de patches, cópias de segurança testadas, registro de acesso e procedimento de resposta a incidente. A imagem Docker e dependências devem ser versionadas e verificadas antes de publicação. Não conectar o endpoint HTTP diretamente ao barramento de controle.

Qualquer integração futura com OT deve passar por engenharia de segurança de máquina e cibersegurança, análise de perigos e validação com o fabricante/integrador do sistema de controle. Ações de redução de velocidade ou parada não devem ser comandadas por esta API. A parada de emergência deve permanecer em sistema de segurança projetado e validado para essa função. As recomendações de segmentação e controles seguem a orientação de OT do [NIST SP 800-82 Rev. 3](https://csrc.nist.gov/pubs/sp/800/82/r3/final); a revisão 4 ainda é rascunho em outubro de 2026.

## Sustentabilidade e ESG

Redução de refugo, retrabalho, consumo de energia por peça aprovada e perdas por parada são hipóteses de benefício, não resultados comprovados neste projeto. O OEE preditivo e as probabilidades do modelo não demonstram, isoladamente, redução de emissões ou consumo.

Para medir benefício, definir antes de um piloto: período-base; volume e família de produto; material; turno e condição de máquina; numerador e denominador para refugo e retrabalho; energia medida por lote/unidade conforme; e regra de atribuição ao sistema. Comparar períodos equivalentes, registrar incerteza e mudanças de mix, e preservar evidência de inspeção e decisão. Publicar ganho apenas após comparação observada e reprodutível. Não estimar CO₂ evitado sem fator de emissão, fronteira e método registrados.

## Condições para uma próxima etapa industrial

1. Aprovar e versionar a tabela de critérios operacionais antes de qualquer ação automática ou recomendação final.
2. Obter rótulos de inspeção com fonte/responsável e protocolo; manter os atuais `NAO_CLASSIFICADA` até lá.
3. Corrigir e reavaliar o pipeline temporal sem estatísticas de teste no pré-processamento, e comprovar inferência online com as mesmas features.
4. Fornecer imagens térmicas com calibração, escala e parâmetros antes de habilitar medição HAZ.
5. Definir controlador/operadores, base legal quando houver dados pessoais, retenção, acesso e resposta a incidentes.
6. Concluir teste e análise de perigos/cibersegurança antes de qualquer integração OT.
7. Medir uma linha de base de refugo, retrabalho, OEE e energia para sustentar alegações ESG.

## Referências normativas e técnicas

- Brasil. Lei nº 13.709/2018 (LGPD), texto compilado, link acima.
- ANPD. Guia Orientativo para Definições dos Agentes de Tratamento e do Encarregado, versão publicada em 2024, link acima.
- União Europeia. Regulamento (UE) 2024/1689, versão consolidada no EUR-Lex, e alteração (UE) 2026/1744, links acima.
- União Europeia. Regulamento (UE) 2023/1230 sobre máquinas, resumo oficial EUR-Lex, link acima.
- NIST. SP 800-82 Rev. 3, Guide to Operational Technology (OT) Security, setembro de 2023, link acima.
- VulcanTech. `Projeto Integrador Avançado certoo.pdf`, `Prática tabela.xlsx`, código em `src` e artefatos existentes em `Docs/Entregaveis/ML`.
