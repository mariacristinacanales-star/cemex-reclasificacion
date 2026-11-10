"""CEMEX: limpieza (notebook 1), outliers + transformación (notebook 2),
clasificación RF+SMOTE (notebook 5). No es un modelo preentrenado verificado.
"""
import io
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title='Reclasificación CEMEX', page_icon='📄')
st.title('Reclasificación automática de costos CEMEX')
st.caption('Carga un Excel, aplica la preparación de los notebooks 1 y 2 y genera un archivo con categorías propuestas mediante Random Forest + SMOTE del notebook 5.')

FEATURES = [
    'Denom.clase de coste_num', 'Cl.coste_num', 'Jerarquía FOP_num',
    'Costo Estandar_num', 'Cl.doc_num', 'Descrip.clases coste_num'
]
ENCODINGS = [
    'Segmento', 'Cl.coste', 'Denom.clase de coste', 'Denom.obj.',
    'Descrip.clases coste', 'Usuario', 'Cl.doc', 'Clasificación',
    'Jerarquía FOP', 'Nombre UO', 'Plaza', 'Costo Estandar',
    'Zona', 'Cia', 'Llave', 'Tipo de registro', 'Categoría de costo'
]
REQUIRED = [
    'Actual', 'Denom.obj.', 'Descrip.clases coste', 'Nº doc.',
    'Usuario', 'Nombre UO', 'Plaza', 'Cl.doc', 'Denom.obj.interl.',
    'Material', '    Ctd.reg.', 'UM', 'Costo Estandar',
    'Denom.clase de coste', 'Cl.coste', 'Jerarquía FOP'
]
DROPS_1 = [
    'Database OG ', 'Texto doc.', 'Ref.anul.', 'Nº pers.', 'Año',
    'Fecha contable', 'MonT', 'Doc.compr.', 'Texto de pedido',
    'Ord. partner', 'Denominación', 'Texto breve de material'
]


def contains(df, col, pattern):
    return df[col].astype('string').str.contains(pattern, case=False, na=False, regex=True)


def etiquetar_segun_reglas(df):
    """Mismo orden y precedencia de las celdas 62 a 71 del notebook 1."""
    df['Categoría de costo'] = pd.Series(pd.NA, index=df.index, dtype='object')
    def set_cat(condition, name):
        df.loc[condition, 'Categoría de costo'] = name

    set_cat(df['Denom.clase de coste'].isin([
        '5604 MATERIAS PRIMAS','5605 SUMINISTROS PRO','5610 REVENTA DE PROD'
    ]), 'Materiales y materias primas')
    set_cat(contains(df,'Descrip.clases coste',r'Aditivos|Agregado|Cemento|Material|Papeleria|Papelería|Oficina'), 'Materiales y materias primas')
    set_cat(contains(df,'Costo Estandar',r'Elem Seg') | contains(df,'Descrip.clases coste',r'segur'), 'Elementos de seguridad')
    set_cat(df['Denom.clase de coste'].isin(['5681 COMBUSTIBLES','5603 ENERGETICOS','5602 ENERGIA ELECTR.']), 'Combustibles y energía')
    set_cat(contains(df,'Costo Estandar',r'Gasolina|Combustible|Diesel|Energia|Energía'), 'Combustibles y energía')
    set_cat(contains(df,'Descrip.clases coste',r'Diesel'), 'Combustibles y energía')
    set_cat(df['Denom.clase de coste'].isin(['5619 MANTENIMIENTO','5619 MAINTENANCE']), 'Mantenimiento y gastos operativos')
    set_cat(contains(df,'Costo Estandar',r'Mtto|Mantenimiento'), 'Mantenimiento y gastos operativos')
    set_cat(contains(df,'Descrip.clases coste',r'Cambio de Inventarios|Gastos Complementarios Mineros'), 'Mantenimiento y gastos operativos')
    set_cat(df['Denom.clase de coste'].isin(['5689 EQPS/SERVS.TELE','5673 SERVICIOS','5617 OTROS SERVICIOS','5688 EQPO.COMPUTO/SW']), 'Servicios y tecnología')
    set_cat(contains(df,'Descrip.clases coste',r'GPS|telecom|celular|software|comput'), 'Servicios y tecnología')
    set_cat(df['Denom.clase de coste'].isin([
        '5630 SLDOS./SALARIOS','5630 OTS.SUELDOS/SAL','5630 OTS.PRESTAC.SOC',
        '5630 SEGURIDAD SOCIA','5630 INDEMNIZACIONES','5630 PREST.NO.GARANT',
        '5672 HONOR.PERS.NATU','5671 HONOR.PERS.JURI'
    ]), 'Personal y prestaciones')
    set_cat(contains(df,'Costo Estandar',r'Sueldos|Personal'), 'Personal y prestaciones')
    set_cat(df['Denom.clase de coste'].eq('5676 GASTOS DE VIAJE'), 'Viajes y movilidad')
    set_cat(contains(df,'Costo Estandar',r'Gastos de Viaje'), 'Viajes y movilidad')
    set_cat(contains(df,'Descrip.clases coste',
        r'Diversos|Cafeteria|Cafetería|Comisiones y Regalias|Reuniones de Trabajo|Aseo y Limpieza|'
        r'Relac. Publicas|Relac. Públicas|Festejos y Regalos|Stand Eventos y Convenciones|'
        r'Afiliaciones|Gastos Legales|Articulos Promocionales|Suscripciones y Publicaciones|'
        r'Indemnizaciones por Calidad|Alimentos y Gastos Zootecnia'),
        'Gastos administrativos y generales')
    set_cat(df['Denom.clase de coste'].isin(['5678 SEGUROS/FIANZAS','5679 IMPUESTOS']), 'Seguros, impuestos y permisos')
    set_cat(contains(df,'Descrip.clases coste',r'Tramites y Licencias|Trámites y Licencias|Multas y actualiz.de impuestos'), 'Seguros, impuestos y permisos')
    set_cat(df['Denom.clase de coste'].isin([
        '5675 ARRENDAMIENTOS','5675 INTERCO ARREND','5683 AMORTIZACION',
        '5683 AMORTIZATION','5683 PC-AMORTIZATION','5684 PROVISIONES'
    ]), 'Arrendamientos, amortizaciones y provisiones')
    set_cat(contains(df,'Costo Estandar',r'Leasing|Alq|Amort|Provision|Provisión'), 'Arrendamientos, amortizaciones y provisiones')
    return df


def limpiar(df):
    # Notebook 1: 0 -> NaN en todas las columnas, filtro de faltantes, documentos y montos
    df = df.copy()
    df['Fila_excel_original'] = np.arange(len(df)) + 2
    df = df.replace([0, '0'], np.nan)
    df.drop(columns=DROPS_1, errors='ignore', inplace=True)
    df = df.dropna(subset=['Actual','Denom.obj.','Descrip.clases coste','Nº doc.',
                           'Usuario','Nombre UO','Plaza']).copy()
    validos = ['KI','KV','KY','R1','SA','WA','WE']
    cond = df['Cl.doc'].isna() & df['Denom.obj.interl.'].isin(validos)
    df.loc[cond, 'Cl.doc'] = df.loc[cond, 'Denom.obj.interl.'].astype(str)
    df = df.dropna(subset=['Cl.doc']).copy()
    df.drop(columns=['Denom.obj.interl.'], errors='ignore', inplace=True)
    df['Material'] = pd.to_numeric(df['Material'], errors='coerce').astype('Int64')
    cond_drop = (df['Material'].notna() & df['    Ctd.reg.'].isna()) | (df['    Ctd.reg.'].notna() & df['UM'].isna())
    df = df.loc[~cond_drop].copy()
    cond1 = df['Material'].notna() & df['    Ctd.reg.'].notna() & df['UM'].notna()
    cond2 = df['Material'].isna() & df['    Ctd.reg.'].notna() & df['UM'].notna()
    df['Tipo de registro'] = np.select([cond1,cond2],['Material','Producto no material'], default='Servicio u otro')
    df.drop(columns=['Material','    Ctd.reg.','UM'], errors='ignore', inplace=True)
    df['Costo Estandar'] = df['Costo Estandar'].fillna('No especificado')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    df = df.loc[df['Actual'].ge(0)].copy()
    if df.empty: raise ValueError('Ningún registro sobrevivió a los filtros de limpieza del notebook 1.')
    return etiquetar_segun_reglas(df)


def transformar(df):
    # Notebook 2: IQR 1.5, imputación de outliers con la mediana, eliminación y LabelEncoder
    df = df.copy()
    for col in ['Mes','Actual','Ce.coste','Área func.','Nº doc.','CC','CeBe']:
        if col not in df.columns: continue
        s = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3-q1
        mask = (s < q1-1.5*iqr) | (s > q3+1.5*iqr)
        df.loc[mask, col] = s.median()
    df.drop(columns=['Mes','Ce.coste','Área func.','Nº doc.'], errors='ignore', inplace=True)
    map_names = {'Denom.obj.':'Denom.obj._num'}
    for col in ENCODINGS:
        if col not in df.columns: continue
        encoded_col = map_names.get(col, col+'_num')
        values = df[col].astype('string').fillna('nan').astype(str)
        df[encoded_col] = LabelEncoder().fit_transform(values)
    if not set(FEATURES).issubset(df.columns):
        raise ValueError('Faltan columnas transformadas para el modelo: '+', '.join(set(FEATURES)-set(df.columns)))
    return df


def clasificar(df):
    # Notebook 5: RF + SMOTE (random_state=0).
    # Entrena con filas previamente etiquetadas por las reglas del notebook 1.
    etiquetas = df['Categoría de costo'].notna()
    X = df.loc[etiquetas, FEATURES]
    y = df.loc[etiquetas, 'Categoría de costo'].astype(str)
    if y.nunique() < 2: raise ValueError('Las reglas identificaron menos de dos categorías. No hay datos suficientes para entrenar el modelo.')
    conteos = y.value_counts()
    if conteos.min() < 2:
        # sin al menos dos registros minoritarios, SMOTE no se puede aplicar
        raise ValueError('Al menos una categoría tiene menos de 2 filas. SMOTE no puede entrenarse con estos datos.')
    k = min(5, int(conteos.min())-1)
    X_bal, y_bal = SMOTE(random_state=0,k_neighbors=k).fit_resample(X,y)
    modelo = RandomForestClassifier(random_state=0, n_jobs=-1)
    modelo.fit(X_bal,y_bal)
    preds = modelo.predict(df[FEATURES])
    result = df.copy()
    result['Categoría reclasificada'] = preds
    result['Origen categoría entrenamiento'] = np.where(etiquetas, 'Regla del notebook 1', 'Sin regla aplicable')
    result['Revisar cambio'] = np.where(etiquetas & (df['Categoría de costo'].astype(str)!=preds), 'Sí','')
    return result

# Reconoce diferencias inocuas de escritura sin confundir columnas diferentes.
# No inventa datos ni rellena columnas faltantes.
ALIASES = {
    'Denom.obj.': ['Denom obj', 'Denom.obj', 'Denominación objeto',
                   'Denominacion objeto', 'Denominación del objeto'],
    '    Ctd.reg.': ['Ctd.reg.', 'Ctd reg', 'Ctd.reg',
                       'Cantidad registrada', 'Cantidad registro'],
    'Nº doc.': ['N° doc.', 'Nº doc', 'No doc', 'Número de documento'],
    'Costo Estandar': ['Costo Estándar'],
}


def normalizar_nombre(nombre):
    nombre = unicodedata.normalize('NFKD', str(nombre))
    nombre = ''.join(c for c in nombre if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '', nombre.casefold())


def detectar_columnas(df):
    # Solo se asigna automáticamente si hay una coincidencia única.
    existentes = list(df.columns)
    usados = set()
    renombrar = {}
    for canonico in REQUIRED:
        if canonico in existentes:
            usados.add(canonico)
            continue
        variantes = [canonico] + ALIASES.get(canonico, [])
        claves = {normalizar_nombre(v) for v in variantes}
        encontrados = [c for c in existentes
                       if c not in usados and normalizar_nombre(c) in claves]
        if len(encontrados) == 1:
            renombrar[encontrados[0]] = canonico
            usados.add(encontrados[0])
    return renombrar


archivo = st.file_uploader('Sube tu Excel original (.xlsx) o CSV',
                           type=['xlsx', 'xls', 'csv'])

if archivo is not None:
    try:
        raw = archivo.getvalue()
        if archivo.name.lower().endswith('.csv'):
            original = pd.read_csv(io.BytesIO(raw), low_memory=False)
        else:
            original = pd.read_excel(io.BytesIO(raw))

        renombres = detectar_columnas(original)
        pendientes = [c for c in REQUIRED
                      if c not in original.columns and c not in renombres.values()]

        # Permitir resolver encabezados realmente distintos sin suponer equivalencias.
        opciones_usadas = set(renombres) | {c for c in REQUIRED if c in original.columns}
        manual = {}
        if pendientes:
            st.warning('Hay columnas que no se pudieron identificar automáticamente. '
                       'Selecciona su equivalente real en tu Excel antes de procesar.')
            with st.expander('Relacionar columnas del Excel', expanded=True):
                for canonico in pendientes:
                    disponibles = ['— Seleccionar columna —'] + [
                        str(c) for c in original.columns if c not in opciones_usadas
                    ]
                    elegido = st.selectbox(f'¿Qué columna corresponde a «{canonico}»?',
                                           disponibles, key='map_'+canonico)
                    if elegido != '— Seleccionar columna —':
                        manual[elegido] = canonico
                        opciones_usadas.add(elegido)

        faltan = [c for c in REQUIRED
                  if c not in original.columns
                  and c not in renombres.values()
                  and c not in manual.values()]

        if st.button('Limpiar, transformar y reclasificar',
                     type='primary', disabled=bool(faltan)):
            st.session_state.pop('csv', None)
            try:
                with st.spinner('Limpiando, transformando y entrenando el modelo...'):
                    preparado = original.rename(columns={**renombres, **manual})
                    limpio = limpiar(preparado)
                    transformado = transformar(limpio)
                    resultado = clasificar(transformado)
                    out_csv = resultado.to_csv(index=False).encode('utf-8-sig')
                    st.session_state['csv'] = out_csv
                    st.session_state['input_id'] = (archivo.name, len(raw))
                st.success('Archivo clasificado. Descarga el CSV.')
            except Exception as err:
                st.error(f'No se pudo procesar el archivo: {err}')

        if 'csv' in st.session_state and st.session_state.get('input_id') == (archivo.name, len(raw)):
            st.download_button('Descargar base reclasificada (CSV)',
                               data=st.session_state['csv'],
                               file_name='CEMEX_base_reclasificada.csv',
                               mime='text/csv', type='primary')

    except Exception as err:
        st.error(f'No se pudo leer el archivo: {err}')

st.caption('Solo se emparejan automáticamente nombres equivalentes sin ambigüedad. '
           'Si falta una columna, debes indicar su correspondencia real. '
           'Las predicciones requieren validación y las filas filtradas en la limpieza '
           'no aparecen en la descarga.')
