"""
Calculadora de salários - Machine Learning (REGRESSÃO)

Aplicação Streamlit que usa um modelo de REGRESSÃO
para predizer salários específicos de profissionais de dados.
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configurações da página
st.set_page_config(
    page_title='Calculadora de salários com ML (Regressão)',
    page_icon='💰',
    layout='wide'
)

@st.cache_resource
def carregar_modelo():
    # Principal parte do código: une o modelo com o aplicativo Streamlit
    try:
        with open('modelo_salarios.pkl', 'rb') as f:
            modelo_completo = pickle.load(f)
        return modelo_completo
    except FileNotFoundError:
        st.error('❌ Modelo não encontrado!')
        st.stop()

def fazer_predicao(dados_usuario, modelo_completo):
    # Cria o dataframe com os dados do usuário
    df_usuario = pd.DataFrame([dados_usuario])

    # Aplica label encoding usando os encoders salvos
    for coluna, encoder in modelo_completo['label_encoders'].items():
        if coluna in df_usuario.columns:
            try:
                # Se o valor não foi visto durante o treinamento, usar o primeiro valor conhecido
                if dados_usuario[coluna] in encoder.classes_:
                    df_usuario[coluna] = encoder.transform([dados_usuario[coluna]])[0]
                else:
                    df_usuario[coluna] = 0 # Valor padrão
            except:
                df_usuario[coluna] = 0

    # Fazer predição (agora retorna um valor numérico)
    salario_predito = modelo_completo['modelo'].predict(df_usuario)[0]

    # Para regressão, podemos calcular uma estimativa de incerteza usando várias árvores
    if hasattr(modelo_completo['modelo'], 'estimators_'):
        # Predições de todas as árvores individuais
        predicoes_arvores = [arvore.predict(df_usuario)[0] for arvore in modelo_completo['modelo'].estimators_]
        std_predicao = np.std(predicoes_arvores)
        intervalo_confianca = 1.96 * std_predicao # ~95% de confiança
    else:
        std_predicao = 0
        intervalo_confianca = 0

    return salario_predito, std_predicao, intervalo_confianca

def formatar_salario(valor):
    """Formata o valor para uma exibição em R$"""
    return f'R$ {valor:,.2f}'.replace(',','X').replace('.', ',').replace('X', '.')

def main():
    # Título e descrição
    st.title("💰 Calculadora de Salários com Machine Learning (Regressão)")
    st.markdown("""
    Esta aplicação usa um modelo de **Random Forest Regressor** treinado com dados reais de profissionais da área de dados para predizer salários específicos em R$.
                
    **Como funciona:** Insira suas informações nos campos abaixo e o modelo fará uma predição do seu salário específico com base em padrões aprendidos dos dados.
    """)

    # Carrega modelo
    modelo_completo = carregar_modelo()

    # Mostra informações do modelo
    if 'metricas' in modelo_completo:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE (Erro Quadrático)", formatar_salario(modelo_completo['metricas']['rmse']))
        with col2:
            st.metric("MAE (Erro Absoluto)", formatar_salario(modelo_completo['metricas']['mae']))
    
    st.sidebar.header("📊 Informações do Profissional")
    st.sidebar.markdown("Preencha seus dados para obter a predição:")

    # Cria formulário para entrada de dados
    with st.sidebar:
        # Dados demográficos
        st.subheader("👤 Dados Pessoais")
        genero = st.selectbox(
            "Gênero",
            options=['Masculino', 'Feminino', 'Prefiro não informar', 'Outro']
        )
        
        etnia = st.selectbox(
            "Etnia",
            options=['Branca', 'Parda', 'Preta', 'Amarela', 'Indígena', 'Prefiro não informar', 'Outra']
        )
        
        idade = st.slider("Idade", min_value=18, max_value=65, value=30)
        
        uf_residencia = st.selectbox(
            "Estado de Residência",
            options=['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'GO', 'DF', 'BA', 'PE', 'CE', 'ES', 'PB', 'MT', 'MS', 'AL', 'RN', 'SE', 'AM', 'AC', 'AP', 'RO', 'RR', 'TO', 'MA', 'PI']
        )
        
        # Dados profissionais
        st.subheader("🎓 Formação")
        nivel_ensino = st.selectbox(
            "Nível de Ensino",
            options=[
                'Graduação/Bacharelado',
                'Pós-graduação', 
                'Mestrado',
                'Doutorado ou Phd',
                'Estudante de Graduação',
                'Ensino Médio',
                'Não tenho graduação formal'
            ]
        )
        
        area_formacao = st.selectbox(
            "Área de Formação",
            options=[
                'Computação / Engenharia de Software / Sistemas de Informação/ TI',
                'Outras Engenharias (não incluir engenharia de software ou TI)',
                'Economia/ Administração / Contabilidade / Finanças/ Negócios',
                'Estatística/ Matemática / Matemática Computacional/ Ciências Atuariais',
                'Ciências Biológicas/ Biomedicina/ Biotecnologia/ Ciências da Vida',
                'Física/ Química/ Geologia/ Ciências Exatas',
                'Ciências Humanas/ Sociais/ Comunicação/ Artes/ Design',
                'Outra opção'
            ]
        )
        
        # Dados de carreira
        st.subheader("💼 Carreira")
        situacao_trabalho = st.selectbox(
            "Situação de Trabalho",
            options=[
                'Empregado (CLT)',
                'Empreendedor ou Empregado (CNPJ)',
                'Estagiário',
                'Servidor Público',
                'Autônomo/ Freelancer',
                'Vivo no Brasil e trabalho remoto para empresa de fora do Brasil',
                'Desempregado',
                'Aposentado',
                'Estudante/ Não trabalho'
            ]
        )
        
        cargo_atual = st.selectbox(
            "Cargo Atual",
            options=[
                'Analista de Dados/Data Analyst',
                'Cientista de Dados/Data Scientist',
                'Engenheiro de Dados/Data Engineer/Data Architect',
                'Analista de BI/BI Analyst',
                'Analista de Negócios/Business Analyst',
                'Analytics Engineer',
                'Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas',
                'Head/Líder/Coordenador/Gerente de Dados',
                'Head/Líder/Coordenador/Gerente de TI',
                'Product Manager',
                'Chief Data Officer (CDO)/ Chief Technology Officer (CTO)',
                'Consultor',
                'Professor/ Pesquisador/ Acadêmico',
                'CEO/Diretor/C-Level',
                'Outra Opção'
            ]
        )
        
        tempo_experiencia_dados = st.selectbox(
            "Tempo de Experiência em Dados",
            options=[
                'Não tenho experiência na área de dados',
                'Menos de 1 ano',
                'de 1 a 2 anos',
                'de 3 a 4 anos',
                'de 5 a 6 anos',
                'de 7 a 10 anos',
                'Mais de 10 anos'
            ]
        )
    
    # Botão para fazer predição
    if st.sidebar.button("🎯 Calcular Salário Previsto", type="primary"):
        # Organiza dados do usuário
        dados_usuario = {
            'genero': genero,
            'etnia': etnia,
            'idade': idade,
            'nivel_ensino': nivel_ensino,
            'area_formacao': area_formacao,
            'situacao_trabalho': situacao_trabalho,
            'cargo_atual': cargo_atual,
            'tempo_experiencia_dados': tempo_experiencia_dados,
            'uf_residencia': uf_residencia
        }
        
        # Fazer predição
        try:
            salario_predito, std_predicao, intervalo_confianca = fazer_predicao(dados_usuario, modelo_completo)
            
            # Mostrar resultado
            st.header("🎯 Resultado da Predição")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="💰 Salário Previsto",
                    value=formatar_salario(salario_predito)
                )
                
            with col2:
                if intervalo_confianca > 0:
                    st.metric(
                        label="📊 Margem de Erro (±95%)",
                        value=formatar_salario(intervalo_confianca)
                    )
            
            # Mostra intervalo de confiança
            if intervalo_confianca > 0:
                limite_inferior = max(0, salario_predito - intervalo_confianca)
                limite_superior = salario_predito + intervalo_confianca
                
                st.info(f"""
                **📈 Intervalo de Confiança (95%):**
                Entre {formatar_salario(limite_inferior)} e {formatar_salario(limite_superior)}
                """)
            
            # Análise da predição
            if salario_predito < 3000:
                st.warning("💡 **Dica:** Salário abaixo da média. Considere especialização ou mudança de cargo.")
            elif salario_predito < 8000:
                st.success("✅ **Bom:** Salário dentro da faixa intermediária.")
            elif salario_predito < 15000:
                st.success("🎉 **Excelente:** Salário acima da média!")
            else:
                st.balloons()
                st.success("🏆 **Excepcional:** Salário no topo da categoria!")
            
            # Mostra fatores que mais influenciam
            st.subheader("📊 Fatores Mais Importantes para o Salário")
            
            # Cria gráfico das importâncias das features
            if hasattr(modelo_completo['modelo'], 'feature_importances_'):
                import plotly.express as px
                
                importance_df = pd.DataFrame({
                    'Feature': modelo_completo['features'],
                    'Importância': modelo_completo['modelo'].feature_importances_
                }).sort_values('Importância', ascending=True)
                
                fig = px.bar(
                    importance_df, 
                    x='Importância', 
                    y='Feature',
                    orientation='h',
                    title='Importância das Variáveis no Modelo'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Disclaimer
            st.info("""
            **📝 Importante:** Esta predição é baseada em dados históricos e deve ser usada apenas como referência. 
            Fatores como performance individual, localização específica, benefícios, tamanho da empresa e outras 
            variáveis não capturadas pelo modelo podem influenciar significativamente o salário real.
            """)
            
        except Exception as e:
            st.error(f"❌ Erro ao fazer predição: {str(e)}")
    
    # Informações sobre o modelo
    with st.expander("ℹ️ Sobre o Modelo"):
        st.markdown("""
        **Algoritmo:** Random Forest Regressor
        
        **Features utilizadas:**
        - Gênero
        - Etnia  
        - Idade
        - Nível de ensino
        - Área de formação
        - Situação de trabalho
        - Cargo atual
        - Tempo de experiência em dados
        - Estado de residência
        
                 **Métricas de Performance:**
         - **RMSE**: Erro médio quadrático das predições (em R$)
         - **MAE**: Erro absoluto médio das predições (em R$)
        
        **Dataset:** Dados de profissionais da área de dados no Brasil (2024)
        
        **Objetivo:** Demonstrar conceitos básicos de machine learning supervisionado (regressão)
        e integração com Streamlit para criação de aplicações interativas.
        """)

if __name__ == "__main__":
    main() 
