import os
import numpy as np
import ollama
import json
import time

def request_embedding_from_ollama(prompt:str="""
    Você é um assistente virtual especializado em sistemas judiciários (como PJe, e-SAJ, Projudi). Sua tarefa é analisar a solicitação do usuário (ticket), resumí-la de forma concisa, e classificá-la em uma de quatro categorias: `erro`, `acesso`, `melhorias`, ou `outros`.

    Sua resposta final deve ser **APENAS** um objeto JSON, sem nenhum texto introdutório. Caso identifique um outro tipo de categoria possível, utilize-a.

    O objeto JSON deve seguir estritamente a seguinte estrutura:
    {
    "analise": "A sentença original completa do usuário.",
    "sentenca_resumida": "Um resumo curto e objetivo que você criou a partir da solicitação do usuário.",
    "thinking_process_message": "Uma explicação passo a passo de como você chegou à classificação. Primeiro, identifique o objetivo central do usuário. Segundo, analise o vocabulário para encontrar palavras-chave (ex: 'não funciona', 'senha', 'sugestão', 'dúvida'). Terceiro, com base na análise, decida a categoria. Por fim, justifique brevemente sua escolha.",
    "categoria": "A categoria final escolhida por você."
    }

    ---
    **DEFINIÇÕES DAS CATEGORIAS:**

    * **erro:** O usuário relata uma falha no sistema. Algo está quebrado, não funciona como esperado ou apresenta uma mensagem de erro. "Erro ao protocolar um processo", "o sistema não gera a petição em PDF".
    
    * **atualização do sistema:** Demanda de atualização de algum sistema. Erro após atualizar o sistema.
        
    * **assinatura digital:** Problema para assinar digitalmente algum processo. Erro ao assinar um documento digitalmente.

    * **problema de conexão:** Problema de acesso à internet e/ou às aplicações internas. Bloqueio de acesso à internet.

    * **carregamento de arquivos:** Erro ao carregar documentos ao chamado.

    * **sistema fora do ar:** Quando algum serviço está fora do ar.
                                  
    * **dados não encontrados:** Erro ao não se encontrar os dados referentes ao chamado/processo judicial.
                                  
    * **alteração de processo:** Erro ao alterar processo, como "alteração do processo judicial"
                                  
    * **consulta de processo:** Erro ao se consultar/visualizar algum processo judicial.

    * **login:** Problema ao acessar a conta. Problema ao se registrar um usuário. Problema ao se alterar dados do usuário.                              
    
    * **acesso:** O usuário tem problemas para entrar ou usar partes do sistema devido a senhas, permissões ou certificados.
        * *Exemplos judiciários:* "Não consigo acessar o PJe com meu certificado digital", "minha senha de advogado expirou", "solicito permissão para visualizar autos sigilosos".

    * **melhorias:** O usuário dá uma sugestão para uma nova funcionalidade ou para aprimorar uma existente.
        * *Exemplos judiciários:* "Sugiro que o sistema permita o upload de vídeos da audiência", "seria bom ter uma busca de jurisprudência mais avançada".

    * **outros:** A solicitação é uma dúvida geral, um pedido de informação, uma questão administrativa ou qualquer outra coisa que não seja um erro, acesso ou melhoria.
        * *Exemplos judiciários:* "Como eu emito uma certidão negativa de débitos?", "qual a competência desta Vara para julgar o caso?".

    """, sentence:str="Sentença do usuário", model_name:str="qwen3:latest"):
    try:
        complete_message = prompt + "\n\n" + sentence.replace(" ", "\n")

        start_time = time.time()        
        # Prepare options for Ollama
        options = {
            'temperature': 0.7,
            'top_p': 1.0,
        }        
        response = ollama.chat(
            model=model_name, 
            messages=[{"role": "user", "content": complete_message}],
            stream=False,
            options=options
        )
        end_time = time.time()  

        return response['message'].get('content', '')
    except Exception as e:
        print(f"Erro ao chamar o modelo Ollama: {e}")
        return None

