import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import pymysql

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_table

# Connexion database
db_user = os.getenv('IMPRESSO_MYSQL_USER')
db_host = os.getenv('IMPRESSO_MYSQL_HOST')
db_name = os.getenv('IMPRESSO_MYSQL_DB')
db_password = os.getenv('IMPRESSO_MYSQL_PWD')
db_url = f'mysql://{db_user}:{db_password}@{db_host}/{db_name}?charset=utf8'
engine = create_engine(db_url, poolclass=NullPool)

# Issue data
data = pd.read_sql('SELECT impresso.issues.id, impresso.issues.year, impresso.issues.month, impresso.issues.day, impresso.issues.newspaper_id, impresso.newspapers.title FROM impresso.issues, impresso.newspapers WHERE impresso.newspapers.id = impresso.issues.newspaper_id ORDER BY impresso.issues.year ASC, impresso.issues.month ASC, impresso.issues.day ASC', engine)

data['datetime'] = pd.to_datetime(data.year*10000+data.month*100+data.day,format='%Y%m%d')

# Make prediction
def get_prediction():
    if(len(NEWS.index) > 2):
        try:
            dfP = pd.read_csv(IDNEWS + '_prediction.csv')
            dfP.datetime = pd.to_datetime(dfP.datetime)
        except FileNotFoundError:
            v = np.array(NEWS.intervalle.values)
            A = np.transpose([v[1:-1],v[2:]])
            b = v[0:-2]
            a = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)), np.transpose(A)),b)
            v_estim = np.convolve(a,b)
            dates = NEWS.datetime[:-2]
            dfP = pd.DataFrame([v[:-2], v_estim[1:], np.absolute(v[:-2]-v_estim[1:]), dates])
            dfP = dfP.transpose()
            dfP.columns = ['signal', 'prediction', 'error', 'datetime']
            dfP.error = pd.to_numeric(dfP.error)
            dfP.to_csv(IDNEWS + '_prediction.csv')
        return dfP
    return None

# Get newspaper
def get_newspaper(ID):
    global IDNEWS
    global NEWS
    global NEWSerror
    if(ID != IDNEWS):
        NWS = data[data.newspaper_id == ID]
        NWS.index = pd.DatetimeIndex(NWS.datetime).to_period('D')
        idx = pd.DatetimeIndex(NWS.datetime).to_period('D')
        prev = idx.append(pd.PeriodIndex([NWS.index[-1]]))
        nxt = idx.insert(0,NWS.index[0])
        interval = prev-nxt
        NWS = NWS.reset_index(drop=True)
        NWS['intervalle'] = interval[:-1].array
        NWS['intervalle'] = NWS['intervalle'].map(lambda x: int(x.nanos/(1e9*60*60*24)))
        NWS['datetime'] = pd.to_datetime(NWS.year*10000+NWS.month*100+NWS.day,format='%Y%m%d')
        IDNEWS = ID
        NEWS = NWS
        NEWSerror = get_prediction()
    return NEWS

# Week indicator
bubbleStyleWhite = dict(
    borderRadius='40px',
    display='block',
    float='left',
    marginRight='6px',
    fontSize='8px',
    padding='1px 4px',
    background='white',
    border='1px solid black',
    color='black'
)

bubbleStyleBlack = dict(
    borderRadius='40px',
    display='block',
    float='left',
    marginRight='6px',
    fontSize='8px',
    padding='1px 4px',
    background='#2d2d2d',
    border='1px solid #2d2d2d',
    color='white'
)

def getDayStyle(weekday,n):
    if n == weekday:
        return bubbleStyleWhite
    else:
        return bubbleStyleBlack

def get_weekInfos(row):
    datetime = row['datetime']
    weekday = datetime.weekday()
    return [
                html.Span(style=getDayStyle(weekday,0), children='M'),
                html.Span(style=getDayStyle(weekday,1), children='T'),
                html.Span(style=getDayStyle(weekday,2), children='W'),
                html.Span(style=getDayStyle(weekday,3), children='T'),
                html.Span(style=getDayStyle(weekday,4), children='F'),
                html.Span(style=getDayStyle(weekday,5), children='S'),
                html.Span(style=getDayStyle(weekday,6), children='S'),
                html.Span(style=styleCalBox(weekday),className='calend',children=[
                    html.Span(style=styleCalMonth,className="calMonth",children=[month[row['datetime'].month]]),
                    html.Span(
                        className='calDay',
                        children=[row['datetime'].day]
                    )]),
                html.I(
                    className='fa fa-caret-down',
                    style=styleCalArrow(weekday)
                )
                
            ]

def get_datimeInfo(infos):
    return [html.A(style=styleJournal,children=[
                html.Img(src='https://dhlabsrv17.epfl.ch/iiif_impresso/'+row['id']+'-p0001/full/400,/0/default.jpg',width='100%'),
                html.Span(style=dict(display='inline-block', position='relative'),children=get_weekInfos(row))],
                   href=row['lien'],target='_blank') if index%5 != 0 else html.A(style=styleJournal5,children=[
                html.Img(src='https://dhlabsrv17.epfl.ch/iiif_impresso/'+row['id']+'-p0001/full/400,/0/default.jpg',width='100%'),
                html.Span(style=dict(display='inline-block', position='relative'),children=get_weekInfos(row))
                    ],href=row['lien'],target='_blank')  for index, row in infos.iterrows()
                ]

    
# Initialisation
IDNEWS      = ''
NEWS        = get_newspaper('LCR')
NEWSerror   = get_prediction()
zoomDwn     = 0
zoomUp      = 0
zoomBottom  = 0
zoomTop     = 0

# Texts
daysIntervalExplanation = 'This graphs shows days interval between a previous issue and the next one throught thw years. You can use the selecting tool to get the links of the issues in impresso in the table tab or have a preview of them in the same named tab. Outliers are highlighted in orange. You can control them with the tools on the left.'

errorThresholdExplanation='x[n] = a1*x[n-1]+a2*x[n-2]. This is the model for the linear prediction. The coefficients a1 and a2 are choosen in order to minimize the total error (energy of the error signal). Therefore, the difference of the days interval prediction and the real days interval is this error (in days). You can decide with the threshold slider what is the minimum error you want to highlight (in orange on the big graph) and the maximum error. The small graph under this text is the mean error for each year and the other graph the frequency analyse of the days interval in the zoom range.'

# Styling variables
month       = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
def styleCalBox(weekday):
    rightPosition = [124,101,79,56,36,15,-6]
    return dict(
                border='1px solid black', background='white',
                position='absolute', display='block',
                width='40px', textAlign='center',
                bottom='26px', right=str(rightPosition[weekday])+'px')
def styleCalArrow(weekday):
    rightPosition = [124,101,79,56,36,15,-6]
    return dict(
                position='absolute', display='block',
                bottom='8px', right=str(rightPosition[weekday]+12)+'px',
                fontSize='2em'
            )
styleCalMonth= dict(
                background='#c30202', color='white',
                display='block', fontSize='0.8em')
styleJournal=  dict(
                display='block', float='left',
                position='relative', width='20%',
                textAlign='center')
styleJournal5= dict(
                display='block', float='left',
                position='relative', width='20%',
                textAlign='center', clear='both'
                )

# Dashboard
app = dash.Dash(external_stylesheets=['https://www.w3schools.com/w3css/4/w3.css','https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css'])

d1Titles = pd.DataFrame({
    'ID': data.newspaper_id.unique(),
    'title': data.title.unique()})

app.layout = html.Div(
    style=dict(background='#6d6d6d', minHeight='100hv'), 
    children=[
        html.Div(
                style=dict(
                    width='100%',
                    background='#343a40',
                    padding='5px',
                    borderTop='#ffeb78 2px solid'
                ),
                children=[
                    dcc.Dropdown(
                        id='d1',
                        options=[{'label': row['title'],'value': row['ID']}
                                for index, row in d1Titles.iterrows()],
                        value='LCR', style=dict(width='500px', float='left'),
                        clearable=False,
                        placeholder="Select a newspaper"
                    ),
                    html.Div(
                        id='div1',
                        style=dict(
                            float='left',
                            margin='5px',
                            textShadow='#65623a -1px 1px 0px',
                            color='white'
                        )
                    ),
                    html.Div(
                        style=dict(
                            clear='both'
                        )
                    )
                ]
            ),
        html.Div(
            style=dict(
                margin='5px',
                marginLeft='16px',
                marginRight='16px',
                marginTop='16px',
                color='black',
                textAlign='center',
                borderBottom='1px solid black',
                textTransform='uppercase',
                fontFamily='georgia',
                fontSize='2em'
            ),
            children=['Publication date interval analyzer']
        ),
        html.Div(
            className='w3-row-padding',
            children=[
                html.Div(className="w3-col s8",
                   children=[
                    html.Div(
                        style=dict(
                            border='1px solid #1f77b4',
                            background='#abe2fb',
                            padding='0px 16px',
                            marginTop='16px'
                        ),
                        children=[
                            html.I(
                                className='fa fa-calendar-o',
                                style=dict(
                                    float='left',
                                    margin='10px 16px 10px 0px',
                                    fontSize='30px'
                                )
                            ),
                            html.P(
                                children=daysIntervalExplanation,
                                style=dict(
                                    fontSize='10px',
                                    fontFamily='courier'
                                )
                            )
                        ]
                    ),
                    dcc.Graph(
                        id='g1',
                        style={'paddingTop': '16px', 'height': '80vh'}
                    ),
                    html.Div(
                        children=[
                            dcc.Tabs(
                                children=[
                                    dcc.Tab(label='Table',
                                        children=[
                                            html.Div(
                                                id='dt1',
                                                style=dict(width='100%')
                                            )
                                        ]
                                    ),
                                    dcc.Tab(label='Preview',
                                        children=[
                                            html.Div(
                                                id='dt2',
                                                style=dict(width='100%')
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        style=dict(
                            background='white',
                            clear='both',
                            marginTop='16px',
                            padding='16px',
                            borderTop='#cacaca 1px solid'
                        )
                    )
                ]),
                html.Div(
                    className="w3-col s4",
                    children=[
                        html.Div(
                            style=dict(
                                border='rgb(0, 0, 0) 3px dotted',
                                background='rgb(255, 235, 120)',
                                padding='0px 16px',
                                marginTop='16px'
                            ),
                            children=[
                                html.I(
                                    className='fa fa-superscript',
                                    style=dict(
                                        float='left',
                                        margin='10px 10px 0px 0px',
                                        fontSize='30px'
                                    )
                                ),
                                html.P(
                                    children=errorThresholdExplanation,
                                    style=dict(
                                        fontSize='10px',
                                        fontFamily='courier'
                                    )
                                )
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H2(
                                    children=['Error threshold'],
                                    style=dict(
                                        borderBottom='1px solid #cccccc',
                                        color='#4e4e4e',
                                        fontFamily='georgia'
                                    )
                                ),
                                dcc.RangeSlider(id='s1', count=1,
                                    min=0, max=1, step=0.01, value=[0.2, 0.5], marks={1:'1',0:'1e-4',0.92:'0.5',0.25:'0.001',0.5:'0.01',0.75:'0.1'}),
                                html.P(['Threshold : ', html.Span(id='sLow'), ' and ', html.Span(id='sHigh'), ' days'],
                                      style={
                                          'marginTop':'25px',
                                          'background':'#fde5e5',
                                          'color':'#8e3e3e',
                                          'border':'1px solid #ff4343',
                                          'borderRadius':'3px',
                                          'padding':'6px'
                                      }
                                ),
                                dcc.Graph(id='g4')
                            ],
                            style={'marginTop': '16px', 'background': 'white', 'padding': '16px', 'paddingTop': '1px'}
                    )]
                ),
                dcc.Graph(
                    className="w3-col s4",
                    id='g2',
                    style={'marginTop': '16px'}
                )
            ]
        ),
        html.Div(
            style=dict(height='500px')
        )

	])


# Dash events
@app.callback(
    Output('g1', 'figure'),
    [
        Input('d1', 'value'),
        Input(component_id='s1', component_property='value')
    ])
def update_output(ID, value):
    global NEWS
    global NEWSerror
    
    minErr = 4
    maxErr = 10
    
    if ID is not None:
        NEWS   = get_newspaper(ID)
            
    if value is None:
        NEWS   = get_newspaper(ID)
        layout = {
                    'hovermode':'closest',
                    'margin': {'l': 60, 'r': 0, 'b': 80, 't': 60},
                    'title': 'Days interval between issues',
                    'xaxis' : {
                                'title': 'Time',
                                'range': [NEWS.datetime.min(), NEWS.datetime.max()]
                              },
                    'yaxis' : {
                        'title': 'Day intervals',
                        'range': [NEWS.intervalle.min(), NEWS.intervalle.max()]
                    }
                 }
    else:
        minErr = np.power(10, 5*value[0]-4)*NEWSerror.error.max()
        maxErr = np.power(10, 5*value[1]-4)*NEWSerror.error.max()
        layout = {
                    'hovermode':'closest',
                    'margin': {'l': 60, 'r': 0, 'b': 80, 't': 60},
                    'title': 'Days interval between issues',
                    'xaxis' : {
                        'title': 'Time',
                        'range': [zoomDwn, zoomUp]
                    },
                    'yaxis' : {
                        'title': 'Day intervals',
                        'range': [zoomBottom, zoomTop]
                    }
                 }
        
    if NEWSerror is not None:
        return    {
                        "data": [
                                go.Scattergl(
                                    x = NEWS.datetime, y = NEWS.intervalle,
                                    mode = 'markers', name = 'Intervals'),
                                go.Scattergl(
                                    x = NEWSerror[(NEWSerror.error>minErr)&(NEWSerror.error<maxErr)].datetime,
                                    y = NEWSerror[(NEWSerror.error>minErr)&(NEWSerror.error<maxErr)].signal,
                                    mode = 'markers', name = 'Outliers')
                                ],
                        'layout': layout
                   }
    else:
        return    {
                        "data": [
                                go.Scattergl(
                                    x = NEWS.datetime, y = NEWS.intervalle,
                                    mode = 'markers', name = 'Intervals')
                                ],
                        'layout': layout
                   }

# Dash events
@app.callback(
    Output(component_id='g2', component_property='figure'),
    [
        Input(component_id='d1', component_property='value'),
        Input(component_id='g1', component_property='relayoutData')
    ]
)
def update_output_content(ID, relayoutData):
    global zoomDwn
    global zoomUp
    global zoomBottom
    global zoomTop
    
    x0 = None
    x1 = None
    y0 = None
    y1 = None
    
    if ID is not None:
        get_newspaper(ID)
    
    if relayoutData is not None:
        if relayoutData.get('xaxis.range[0]', None) is not None:
            x0 = relayoutData['xaxis.range[0]']
        if relayoutData.get('xaxis.range[1]', None) is not None:
            x1 = relayoutData['xaxis.range[1]']
        if relayoutData.get('yaxis.range[0]', None) is not None:
            y0 = relayoutData['yaxis.range[0]']
        if relayoutData.get('yaxis.range[1]', None) is not None:
            y1 = relayoutData['yaxis.range[1]']
    
    if(x0 == None):
        if(x1 == None):
            NWS = NEWS
        else:
            NWS = NEWS[NEWS.datetime<=x1]
    else:
        if(x1 == None):
            NWS = NEWS[NEWS.datetime>=x0]
        else:
            NWS = NEWS[(NEWS.datetime>=x0)&(NEWS.datetime<=x1)]
            
    zoomDwn = x0
    zoomUp = x1
    zoomTop = y1
    zoomBottom = y0
    
    return {
                'data':[
                    {
                        'y':NWS.intervalle.value_counts().head(10).index,
                        'x':NWS.intervalle.value_counts().head(10).values,
                        'type': 'bar',
                        'orientation': 'h'
                    }
                ],
                'layout': {
                    'hovermode':'closest',
                    'margin': {'l': 60, 'r': 0, 'b': 70, 't': 60},
                    'title': 'Day intervall frequencies',
                    'xaxis' : { 'title': 'Number of occurrence' },
                    'yaxis' : { 'title': 'Day intervals' }
                }
            }

# Dash events
@app.callback(
    Output(component_id='div1', component_property='children'),
    [Input(component_id='d1', component_property='value')]
)
def update_output_content(value):
    NEWS      = get_newspaper(value)
    return str(len(NEWS.index)) + ' issues '



# Dash events
@app.callback(
    Output(component_id='sLow', component_property='children'),
    [Input(component_id='s1', component_property='value')]
)
def update_output_content(value):
    return round(np.power(10, 5*value[0]-4)*NEWSerror.error.max(), 2)

# Dash events
@app.callback(
    Output(component_id='sHigh', component_property='children'),
    [Input(component_id='s1', component_property='value')]
)
def update_output_content(value):
    return round(np.power(10, 5*value[1]-4)*NEWSerror.error.max(), 2)


# Dash events
@app.callback(
    Output('g4', 'figure'),
    [Input('d1', 'value')])
def update_output_content(value):
    NEWS   = get_newspaper(value)
    return {
            'data':[
                {
                    'y':NEWSerror.error.groupby(NEWSerror.datetime.dt.year).mean().values,
                    'x':NEWSerror.error.groupby(NEWSerror.datetime.dt.year).mean().index
                }
            ],
            'layout': {
                'height':250,
                'hovermode':'closest',
                'margin': {'l': 30, 'r': 0, 'b': 20, 't': 60},
                'title': 'Error mean'
            }
    }

@app.callback(
    Output('dt1', "children"),
    [Input('g1', 'selectedData')])
def update_output_content(selectedData):
    if selectedData is not None:
        pointNb = list()
        for point in selectedData['points']:
            if point['curveNumber'] == 0:
                pointNb.append(point['pointNumber'])
        infos = NEWS[NEWS.index.isin(pointNb)]
        infos['lien'] = infos['id'].apply(lambda a : 'https://impresso-project.ch/alpha/#/issue/'+a)
        infos.drop(columns=['year','month','day','newspaper_id','title']).to_dict("rows")
        
        tableContent = [html.Tr([
                html.Th(children=['id'],style=dict(borderBottom='1px solid #c7c7c7',padding='4px 0')),
                html.Th(children=['datetime'],style=dict(borderBottom='1px solid #c7c7c7')),
                html.Th(children=['interval'],style=dict(borderBottom='1px solid #c7c7c7')),
                html.Th(children=['lien'],style=dict(borderBottom='1px solid #c7c7c7'))])]+ [html.Tr(
                [
                    html.Td(children=[row['id']],style=dict(borderBottom='1px solid #c7c7c7',padding='4px 0')),
                    html.Td(children=[row['datetime']],style=dict(borderBottom='1px solid #c7c7c7')),
                    html.Td(children=[row['intervalle']],style=dict(borderBottom='1px solid #c7c7c7')),
                    html.Td(children=[html.A(href=row['lien'],target='_blank',children=[row['lien']])],style=dict(borderBottom='1px solid #c7c7c7'))
                ]) for index, row in infos.iterrows()]
        
        return html.Table(children=tableContent,style=dict(width='100%',fontFamily='courier',textAlign='left',marginTop='10px'))
    else:
        return ''

@app.callback(
    Output('dt2', "children"),
    [Input('g1', 'selectedData')])
def update_output_content(selectedData):
    if selectedData is not None:
        pointNb = list()
        for point in selectedData['points']:
            if point['curveNumber'] == 0:
                pointNb.append(point['pointNumber'])
        infos = NEWS[NEWS.index.isin(pointNb)]
        infos['lien'] = infos['id'].apply(lambda a : 'https://impresso-project.ch/alpha/#/issue/'+a)
        infos.drop(columns=['year','month','day','newspaper_id','title']).to_dict("rows")
        infos = infos.reset_index(drop=True)
        return get_datimeInfo(infos)
    else:
        return ''
    

    

if __name__ == '__main__':
    app.run_server(debug=True, host="128.178.115.26",port=9102)