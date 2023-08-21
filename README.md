# SentenceTransforms
Uso de modelo com sentence transforms para busca semantica/// AI model for semantic search
O modelo é um esqueleto de um projeto para realizar busca semantica com banco de dados, utilizando o modelo(all-MiniLM-L6-v2).
 Nesse rascunho é utilizado poucos dados para melhor vizualização e entendimento dos embeddings 
 Dentro desse rascunho foi colocado um sistema de filtros para melhor obtenção dos resultados, fazendo com que assim sejam eliminados resultados os quais não nos interessam
Dentro da lista que simula o banco de dados foi colocado informações referente a cada influenciador retiradas do seu instagram 

________________________________________________________________________________________

 !pip install sentence_transformers

from sentence_transformers import SentenceTransformer, util


class Influenciador:
    def __init__(self, nome, descricao, categorias):
        self.nome = nome
        self.descricao = descricao
        self.categorias = categorias


influenciadores = [
    Influenciador("Thiago nigro", "ações e dinheiro para investir nomercado financeiro e marketing digital", ["Investimento", "finanças"]),
    Influenciador("Tiago Flinch", "comprar ações no mercado financeiro e aprender a investir", ["Investimento", "finanças"]),
    Influenciador("Virginia Fonseca", "Eu tava c taaaaanta saudadinha das minhas totocassssssss 💖💜.Verificado Eles arremataram nossa experiência por R$ 735.000,00! Mt gratidão!! 🥰 Fizeram total diferença na vida de milhares de crianças e famílias que são ajudadas pelo Instituto Neymar Jr.Td honra e glória a Deus 🙌🏻💜💖Pronta pra hoje!!! Muito feliz e grata por ter sido convidada pra fazer parte do time de apresentadores desse evento tão especial e por uma causa tão import/ante, em prol de milhares de crianças e famílias que o @institutoneymarjr ajuda diariamente!!! Que Deus nos abençoe e bora 💖💜Virginia Fonseca CostaArtista2023 é NOSSO 💖💖💜💜Toda honra e glória a Deus 🙌🏻CONTATO: virginia@talismadigital.com.br 📩MELATONINA + TRIPTOFANO!!!", ["Moda", "beleza"]),
    Influenciador("Casimiro", "VivaChargers,vivaLebronJames, viva a Pizza e acima de tudo, viva o Vasco da Gama.🏀🏈❤️⚡️😎NBA All-Star Weekend 🏀✅lado de pessoas fodas! Desculpa não postar tanto eu não tenho costume ai SIM, A GENTE VAI TRANSMITIR O MUNDIAL DE CLUBES FIFA! 😍A CazéTV não para! A gente avisou que teria mais novidades. E agora temos o orgulho de contar que vamos transmitir o Mundial de Clubes FIFA AO VIVO, COM IMAGENS e DE GRAÇA! E o melhor: vai ter cobertura direto do Marrocos, hein? Daquele nosso jeito que vocês já conhecem!Depois da emoção da Copa do Mundo, vai ser incrível levaroMundial de Clubes FIFA para todo o Brasil na CazéTV! Será que vem aí Real Madrid x Flamengo? 👀esqueço…☹️😂VerificadoA gente continua realizando sonhos na CazéTV! E vamos começar 2023 juntos no Campeonato Carioca! TODOS os jogos de Botafogo e Vasco como MANDANTES, com imagens e de graça, no YouTube e na Twitch! Vem com a gente fazerFigura pública@twitch | @youtube📧casimiro@livemode.nettwitch.tv/casimito história de novo! 🔥💢", ["Esportes", "futebol"]),
]


def find_most_similar(user_text, user_category):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    influencers_in_category = [inf for inf in influenciadores if user_category.lower() in map(str.lower, inf.categorias)]

    if not influencers_in_category:
        return []

    corpus = [inf.descricao for inf in influencers_in_category]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    query_embedding = model.encode([user_text], convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings)[0]

    sorted_influencers = [(influencers_in_category[hit['corpus_id']].nome, hit['score']) for hit in hits]

    return sorted_influencers


texto_usuario = input("Por favor, digite suas palavras-chave: ")
categoria_solicitada = input("Por favor, escolha uma categoria: ")

influenciadores_similares = find_most_similar(texto_usuario, categoria_solicitada)

if influenciadores_similares:
    print('\nInfluenciadores similares encontrados, ordenados por similaridade:\n')
    for influencer, score in influenciadores_similares:
        print(f'Nome: {influencer}\nScore: {score}\n{"-"*50}\n')
else:
    print(f'Nenhum influenciador encontrado para a categoria {categoria_solicitada}')
