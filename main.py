#!/usr/bin/env python
# coding: utf-8

# # People
# 
# What are they? How can we represent people in a meaningful way without having *real* data to draw from. In our organization we should have a few key pieces of information on each person:
# 
# * Unique identifier (a name would be great)
# * Office/Location
# * Job Title
# * Team / Line of Business
# * Manager (this is key to understanding how reporting works)
# * Other info: programming languages, apps used, timezone, date hired, projects worked on, ...
# 
# ## Faker
# 
# Let's generate some fake data! https://faker.readthedocs.io/en/master/

# In[229]:


from faker import Faker
fake = Faker()

fake.profile() # lots of great stuff in here!


# We can also do things like ensure uniqueness for individual entries across all entries

# In[230]:


from faker.exceptions import UniquenessException
try:
    for i in range(10):
        print(fake.unique.prefix())
except UniquenessException:
    print("😟")


# Try generating a few people and see if it looks like a good representation of our organization

# In[231]:


...


# This is a good start but ... it's kind of wonky. We have people all over the world with so many different jobs! Let's keep the spirit of this but implement some of our own limitations on fields to ensure things line up with what we'd expect a company org to look like
# 

# First, a few more interesting features: we can also register new `providers` if anything is missing. If needed these can be customized for different locales 

# In[232]:


from faker.providers import DynamicProvider

employment_status_provider = DynamicProvider(
     provider_name="employment",
     elements=["Full Time", "Part Time", "Contract"],
)

fake.add_provider(employment_status_provider)

fake.employment()


# We can customize this further by using the `Faker.BaseProvider`

# In[233]:


# first, import a similar Provider or use the default one
from more_itertools import one
from faker.providers import BaseProvider

# create new provider class
class EmploymentStatus(BaseProvider):
    statuses = {"Full Time": 0.7, "Part Time": 0.05, "Contract": 0.3}
    def employment(self) -> str:
        return one(fake.random.choices(
            list(self.statuses), 
            weights=self.statuses.values()
        ))

# then add new provider to faker instance
fake.add_provider(EmploymentStatus)

fake.employment()


# ### A Tech Focused Person Data

# To ground us in this task, let's define a new `Person` object that we can fill up with info (and a few other objects):

# In[234]:


from dataclasses import dataclass, field
from typing import Literal
from enum import Enum, auto
import datetime

class timezone(str, Enum):
    EST = auto()
    PST = auto()
    UTC = auto()

@dataclass
class Location:
    city: str
    tz: timezone
    country: str

@dataclass
class Person:
    """Someone who works in our company!"""
    name: str
    hire_date: datetime.date
    status: Literal["Full Time", "Part Time", "Contract"]
    languages: list[str] = field(default_factory=list)
    manager:str = None
    team: str = None 
    title: str = None
    location: Location = None


# In[235]:


Person(name="Employee #1",hire_date=datetime.date.today(), status="Full Time", location=Location("New York", "EST", "USA"))


# In[236]:


import numpy as np
import random

def choose_a_few(
    options: list[str],
    weights: list[int | float] = None,
    max_choices: int = None,
    min_choices: int = 0,
) -> list[str]:
    """A helpful function to pick a random number of choices from a list of options
    
    By default skews the weights toward the first options in the list"""
    max_choices = np.clip(max_choices or len(options), min_choices, len(options))
    
    # how many choices will we make this time?
    divisor = max_choices * (max_choices + 1) / 2    
    k_weights = [int(x) / divisor for x in range(max_choices, min_choices-1, -1)]
    n_choices = np.random.choice(list(range(min_choices,max_choices+1)), p=k_weights)
    
    # make the choices
    choices = random.choices(options, weights=weights, k=n_choices)
    return list(set(choices))


# Now to make some people. Let's re-use whatever we can from `Faker` and then add some more of our own fields. We can also extend where needed to keep our code clear and consistent:

# In[237]:


class ProgrammingLanguages(BaseProvider):    
    languages = {
        "Python": 0.25,
        "Scala": 0.1,
        "Go": 0.08,
        "JavaScript": 0.3,
        "Java": 0.3,
        "Typescript": 0.17,
        "Erlang": 0.01,
        "Elixir": 0.001,
    }
    def programming_languages(self) -> str:
        return choose_a_few(list(self.languages), weights=self.languages.values())

fake.add_provider(ProgrammingLanguages)


# In[238]:


def make_person() -> Person:
    return Person(
        name = fake.name(),
        hire_date = fake.date_between(start_date="-3y", end_date="today"),
        status = fake.employment(),
        languages = fake.programming_languages(),
        team = None, # hrmmmm this is harder
        title = None, # let's be smarter with this
        location = None, # let's also be smarter with this
    )

make_person()


# Now we can generate more complex attributes in a smart way. Let's set up some rules about where offices are, what teams are in which offices, then pick titles based on other info (e.g. Developers probably know at least one language ... )

# In[239]:


TEAM_TITLES:dict[str,list[str]] = {
    "DevX": ["Engineer", "Engineer", "Engineer", "Engineer", "Engineer", "AVP"],
    "DevOps": ["Engineer", "Senior Engineer", "Manager", "Senior Manager"],
    "Sales": ["Associate", "VP"],
    "Support": ["Analyst", "Manager"],
    "Platform": ["Engineer", "Senior Engineer","Managing Engineer", "AVP", "VP"],
    "Product": ["Engineer", "Manager", "Product Owner", "AVP", "VP"],
    "Internal Tools": ["Engineer", "Senior Engineer", "Manager", "AVP", "VP"],
    "Business": ["Analyst", "Associate", "Vice President", "Director", "Managing Director"]
}

# codify the hierarchical structure
allowed_teams_per_office = {
    "New York": ["Sales", "Product", "Business"],
    "Toronto": ["Platform", "Product", "Internal Tools", "Sales", "Business"],
    "Fort Lauderdale": ["DevX"],
    "Dublin": ["DevOps", "Support"],
    "London": ["Sales", "Business"],
    "Seattle": ["Internal Tools", "Product", "Platform"],
}
offices = {
    location.city: location
    for location in [
        Location("New York", tz="EST", country="USA"),
        Location("Seattle", tz="PST", country="USA"),
        Location("Toronto", tz="EST", country="CAN"),
        Location("London", tz="UTC", country="GBR"),
        Location("Fort Lauderdale", tz="EST", country="USA"),
        Location("Dublin", tz="UTC", country="IRL"),
    ]
}

def title_city_team():
    # just a few locations
    allowed_titles_per_team = TEAM_TITLES
    city = random.choice(list(offices))
    team = random.choice(allowed_teams_per_office[city])
    title = choose_a_few(
        allowed_titles_per_team[team], max_choices=1, min_choices=1
    ).pop()
    
    return {
        "location": Location(city=city, tz=offices[city].tz, country=offices[city].country),
        "title": title,
        "team": team,
    }


title_city_team()


# After running this we should have a better balanced org in terms of region + titles. Then we just need to add the connections in -- i.e. who's the boss?!

# In[240]:


def make_person() -> Person:
    title_city_team_ = title_city_team()
    technical = 1 if "Engineer" in title_city_team_["title"] else 0
    return Person(
        name = fake.name(),
        hire_date = fake.date_between(start_date="-3y", end_date="today"),
        status = fake.employment(),
        languages = fake.programming_languages(),
        **title_city_team_,
    )


# In[241]:


import pandas as pd
people_df = pd.DataFrame((make_person() for _ in range(150)))
people_df.head()


# So, let's group by Team and then pick a manager for everyone. Let's use these rules:
# 
# * People report to someone of a higher title if possible, else to a peer
# * Reporting happens within a team
# * We already ordered `TEAM_TITLES` based on *rank*
# * Team leads should be listed as reporting to themselves (for now)

# In[242]:


# calculate team ranks
ranks = {team: {title: rank + 1 for rank,title in enumerate(titles)} for team, titles in TEAM_TITLES.items()}
for team in ranks:
    people_df.loc[people_df.team==team, "rank"] = people_df.loc[people_df.team==team].title.map(ranks[team])
people_df = people_df.sort_values(by=["team","rank"])
people_df.sample(3)


# In[244]:


# determine supervisor
def naivereportsto(row, df, allow_peer_reports:bool=False):
    supervisor = (
        df[(df.index < row.name)].query(f"""rank > {row["rank"]}""").tail(1)["name"]
    )
    supervisor = supervisor.item() if not supervisor.empty else None
    if not supervisor and allow_peer_reports:
        peer = df[(df.index < row.name)].query(f"""rank  == {row["rank"]}""").head(1)["name"]
        peer = peer.item() if not peer.empty else None
        return supervisor or peer or row["name"]
    return supervisor or row["name"]


def reportsto(df, allow_peer_reports:bool):
    return df.assign(manager=df.apply(naivereportsto, df=df, allow_peer_reports=allow_peer_reports, axis=1))


def supervisors(df, allow_peer_reports:bool):
    df = df.groupby("team", group_keys=False).apply(reportsto, allow_peer_reports=allow_peer_reports).reset_index(drop=True)
    return df


people_df = people_df.pipe(supervisors, allow_peer_reports=True)
people_df.sample(5)


# Now we just need a CEO for all the team leads to report to. Set their manager as themselves to help us out later. We need to make sure to include all the other information in the DF that we just generated, namely `rank` and `manager`. Here let's also set the CEO as reporting to themselves 

# In[245]:


CEO = make_person().__dict__ | {"team":"CEO", "title":"CEO", "status":"Full Time"}
CEO["location"] = CEO["location"].__dict__
people_df = pd.concat([people_df, pd.DataFrame([CEO])])
CEO_mask = people_df.name==CEO["name"]
people_df.loc[(people_df.manager == people_df.name) | CEO_mask ,"manager"]=CEO["name"]
people_df.loc[CEO_mask, "rank"] = people_df["rank"].max()+1


# Alright, we have something now. Does this seems reasonably distributed? Let's use `plotly` to explore our people's dimensions and get a feel for the data

# In[246]:


# let's flatten the nested pieces of the DataFrame (`people_df.location`)
expanded_df = people_df.assign(**people_df.location.apply(pd.Series))
expanded_df


# In[247]:


import plotly.express as px

fig = px.bar(
    expanded_df,
    x="title",
    color="team",
    hover_name="name",
    hover_data=["team", "tz", "city","manager","languages"],
    facet_col="country",
    template="plotly_dark",
)
fig.update_xaxes(matches=None, title_text=None)


# In[ ]:





# # Understanding People with Network Graphs in Python
# 
# Now we can really start

# ## NetworkX
# 
# > NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks - https://networkx.org/
# 
# NetworkX provides an easy-to-use, *fast*, graph framework to represent relationships in-memory

# In[135]:


import networkx as nx

G = nx.Graph()


# It's easy to load in data one at a time

# In[136]:


G.add_node("Me", type="person", languages=["Python"])


# Or from an iterable object

# In[137]:


G.add_nodes_from((
    ("You",dict(languages=["Python","Scala"])),
    ("Them",dict(languages=["Python","Javascript"]))
), type="person")


# Now we can look at the new data structure and play around with it:

# In[138]:


G.nodes() # show all the node labels


# In[139]:


G.nodes(data=True) # show all attributes in nodes


# In[140]:


G.nodes(data="languages") # show a specific attribute


# In[141]:


G.add_edge("Me","You", label="friends") # add an edge connecting two nodes ...
G.add_edge("You","Them", label="friends") # add an edge connecting two nodes ...


# There are lots of common and complex graph analysis functions available to make the most of the data structure 

# In[142]:


nx.shortest_path(G, source="Me", target="Them") # find paths between nodes through edges


# In[143]:


G.adj # find adjacent nodes


# And visualizing is built-in with `matplotlib`. We will pretty this up later with `plotly`

# In[144]:


nx.draw(G, with_labels=True)


# ### Some People Data
# 
# Let's re-use what we made before:

# ## The Graph
# 
# Now we're ready to define and populate the network graph. NetworkX provides helpful methods to populate the structure from an iterable. Here we massage our list of `people` a bit in order to give a unique name to each *node* in the graph:

# In[146]:


G = nx.Graph() # a simple undirected graph
G.add_nodes_from(((person["name"], person) for person in expanded_df.to_dict(orient="records")), person=True, type="person")
G.nodes(data=True) # we can look at the contents (which should be very familiar!)


# ### Visualizing
# 
# Graphs lends themselves well to visual representations. NetworkX also makes this easy to do by tapping into Python's workhorse plotting library, `matplotlib`. We will revisit this later with a more dynamic + interactive approach to visualizing, but for the moment this is the fastest way to get things on paper

# In[148]:


import matplotlib.pyplot as plt
nx.draw(G, with_labels=False)


# Let's add a bit of color to this by mapping colors to the `person.team`. Pick any colorscale from `px.colors` (or make your own!). Generally the *qualitative* colors look nice, anything designed for *categorical* data 

# In[149]:


colors = dict(zip(people_df.team.unique(),px.colors.qualitative.Vivid))
colors


# In[150]:


# here are some helpful helpers to translate colors
def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'
def rgb_string_to_tuple(rgb:str) -> tuple[int,int,int]:
    return tuple(int(c) for c in rgb.replace("rgb(","").replace(")","").split(","))


# Now we can determine what the color should be for each node and pass that into the `nx.draw` call as a list of `node_color`. The easiest way to do this is to use `G.nodes(data=...)` for the attribute you want to extract, which will give you a map from each node to that attribute. `nx` allows you to iterate

# In[151]:


node_colors = [rgb_to_hex(*rgb_string_to_tuple(colors[team])) for _,team in G.nodes(data="team")]
nx.draw(G, node_color=node_colors)


# If this doesn't make much sense yet, it's because we haven't connected any of the nodes together. Adding *edges* to the graph will give shape and meaning to the arrangement of nodes. We can do this similarly to how we added edges. Let's start by connecting nodes by the `manager` attribute.
# 
# Be sure to only add edges that reference nodes that exist

# In[152]:


G.add_edges_from(G.nodes(data="manager"), label="manager", manager=True)


# Now this should look a bit more sensible

# In[153]:


nx.draw(G, node_color=node_colors)


# To view more details of the plot, it's useful to switch to an interactive plotting library like `plotly`. Here we provide a helper function for this, but by no means is this perfect/optimized

# In[154]:


import pandas as pd
import plotly.express as px


# In[155]:


import plotly.graph_objects as go
def px_plot_network_graph_nodes(G:nx.Graph, *, layout=None, **pxkwargs) -> go.Figure:
    # generate the x/y coordinates to represent the graph
    positions = (layout or nx.spring_layout(G))
    # prepare as DataFrame for plotly
    df = pd.DataFrame([{"label": k, "x": v[0], "y": v[1], "size":10, **G.nodes(data=True)[k]} for k,v in positions.items()])
    for column in df.columns[(df.sample(100, replace=True).applymap(type) == set).any(axis=0)]:
        print(f"Coercing column '{column}' to `list`")
        df.loc[~df[column].isna(), column] = df.loc[~df[column].isna(),column].apply(list)
    # handle missing values for size/color parameter
    size = pxkwargs.pop("size", "size")
    df[size] = df[size].fillna(df[size].max())
    color = pxkwargs.get("color")
    df[color] = df[color].fillna(df["type"])
    # create figure
    fig = px.scatter(df, x="x", y="y", hover_data=df.columns, size=size, **pxkwargs)
    fig.update_layout(
        xaxis=go.layout.XAxis(visible=False),
        yaxis=go.layout.YAxis(visible=False)
    )
    return fig


def px_plot_nx(G:nx.Graph, *, layout=nx.spring_layout, with_edges=False, **nodekwargs) -> go.Figure:
    """Draw a graph using ``plotly``

    Kwargs are passed through to `px.scatter` and can be used to control the attributes that 
    map ``color``, ``size``, ``facet_row``, ... to attributes in the graph nodes
    
    Notes
    -----
    Rendering ``with_edges`` is expensive and should be avoided during exploratory plotting
    """
    # Generate positions, edges 
    nodes = layout(G)
    edges = [{
        "x": [nodes[source][0],nodes[target][0]], 
        "y": [nodes[source][1],nodes[target][1]]} for source, target in G.edges()
    ]
    # Plot nodes
    figure = px_plot_network_graph_nodes(G, layout=nodes, **nodekwargs)
    if with_edges: # Add edges to nodes
        figure.add_traces([
            px.line(
                x=edge["x"],
                y=edge["y"],
                color_discrete_sequence=["grey"],
            ).data[0] for edge in edges
        ])
        figure.data = figure.data[::-1] # shuffle edges behind nodes
    return figure


# Use this to make a few plots of the graph and verify that:
# 
# * The reporting structure makes sense
# * The job titles are distributed as you'd expect
# * The locations make sense
# 
# **Note: plotting `with_edges=True` is quite expensive, try toggling it off if you find it bothersome**

# In[130]:


from functools import partial

layout = (
    nx.spring_layout
)  # partial(nx.spring_layout,k=0.1, iterations=20) # or customize how the layout is generated
px_plot_nx(
    G,
    color="country",
    layout=layout,
    with_edges=False,
    hover_name="name",
    size="rank",
    template="plotly_dark",
)  # ,text="label")


# In[131]:


px_plot_nx(
    G,
    color="team",
    layout=layout,
    with_edges=False,
    hover_name="name",
    size="rank",
    template="plotly_dark",
)  # ,text="label")


# In[ ]:





# Now we can take extra info (attributes) of each person (node) and map those onto nodes. This will allow us to connect people *through* common attributes, and not just through relationships like "reporting structure" or "hierarchy"

# In[3]:


from itertools import chain
from more_itertools import always_iterable

def add_nodes_from_attributes(G: nx.Graph, *, attribute:str, default=[], flag:str):
    attributes = {person:attribute for person, attribute in G.nodes(data=attribute, default=default) if attribute}
    for attr in set(chain.from_iterable((always_iterable(value) for value in attributes.values()))):
        if attr:
            G.add_node(attr, **{flag: True, "type":flag})
        
add_nodes_from_attributes(G, attribute="languages", flag="language")
#add_nodes_from_attributes(G, attribute="apps", flag="app")
add_nodes_from_attributes(G, attribute="tz", flag="timezone", default="")


# Let's add the edges in now connecting people to the app/language nodes!

# In[4]:


from more_itertools import always_iterable

def just_people(G):
    return lambda n: G.nodes()[n].get("person")

def add_edges_from_attributes(G:nx.Graph, *, attribute:str, weight:int=1):
    nodes = G.nodes()
    for name, attributes in nx.subgraph_view(G, filter_node=just_people(G)).nodes(data=attribute):
        for attr in always_iterable(attributes):
            if attr in nodes:
                G.add_edge(name, attr, weight=weight, **{attribute:True})


# In[5]:


add_edges_from_attributes(G, attribute="languages")
add_edges_from_attributes(G, attribute="tz")
#add_edges_from_attributes(G, attribute="apps")
#add_edges_from_attributes(G, attribute="manager")


# This should look interesting! We used a **force-directed layout** to draw the graph, meaning that the edges between nodes are **pulling** the nodes together in order until they find an equilibrium point. This also takes into account the weights we applied to edges, with higher weighed edges behaving like springs with higher spring constants

# In[6]:


px_plot_nx(G, height=800, hover_name="label", color="team", size="rank", with_edges=False, template="plotly_dark")


# ## Extras
# 
# For the sake of the tutorial we are going to continue with a visual exploration of the graph, but we can also use computational methods. Here are a few jumping off points:
# 
# Let's use a few methods to investigate how the *people* nodes in the graph are connected and see if we can find clusters and *neighbourhoods*
# 
# We can look at just a "neighbourhood" around a person ... but this is currently unweighted, so we're just seeing anything 1 node away, instead of looking at a more complete view of "closeness"

# ```python
# someone = list(G.nodes)[0]
# ego = nx.ego_graph(G, someone, undirected=True, radius=2)
# ego_people = nx.subgraph_view(ego, filter_node=just_people(G)).nodes()
# nx.draw_networkx(ego, nodelist=ego_people)
# ```

# This information is also easy to recover frbom the Graph itself

# ```python
# peoplenodes = nx.subgraph_view(ego, filter_node=just_people)
# connectivity = nx.all_pairs_node_connectivity(ego, nbunch=peoplenodes)
# connectivity
# ```

# We can also look at where nodes get placed in our force-directed graph and look for closeness there

# ```python
# import numpy as np
# import xarray as xr
# 
# positions = {
#     name: pos for name, pos in nx.spring_layout(G).items() if name in peoplenodes
# }
# positions = xr.DataArray(
#     list(positions.values()),
#     coords={"person": list(positions), "position": ["x", "y"]},
#     dims=["person", "position"],
#     attrs=dict(description="Node locations in a force-directed layout"),
# )
# positions
# ```

# We can now do some matrix math to find the pairwise euclidean distances between each node!

# ```python
# similarities = np.sqrt(((positions - positions.rename(person="person2"))**2).sum("position"))
# similarities.name = "distance"
# similarities
# ```

# ```python
# friends = similarities.sel(person="Lucas").sortby(similarities.sel(person="Lucas"))
# friends[:2].person2
# ```

# But this is the same approach (in spirit) to representing a graph as vectors in N-dimensional space (only here we do just 2-dim). A more common approach is to use `node2vec` and then look for closeness in the vectors

# ## A Very Simple ``Dash`` App

# In[100]:


from dash import Dash, html, Output, Input

app = Dash()

app.layout = html.Div([
    html.H1("A Simple App", id="title"),
    html.P("Structure things like you would in HTML"),
    html.Button("Click me to do something", id="button"),
])

@app.callback(
    Output("title", "children"), # change this value
    Input("button", "n_clicks"), # when this changes
    prevent_initial_call = True
)
def update_title_on_buttonclick(n_clicks):
    print(n_clicks) # sure why not
    return "A Simple App **That Does Things!**"

#app.run()


# ## A Bit More Complicated App
# 
# Let's use bootstrap to spruce this up a bit

# In[101]:


import dash_bootstrap_components as dbc
from dash import dcc

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div(
    [
        dbc.NavbarSimple(brand="Communities in Network Graphs"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Customize Lorems"),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("N Paras: "),
                                        dbc.Input(
                                            value=2, type="number", id="n_paragraphs"
                                        ),
                                    ]
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Color: "),
                                        dcc.Dropdown(
                                            value=None,
                                            options=[
                                                {"label": o, "value": f"var(--bs-{o})"}
                                                for o in [
                                                    "red",
                                                    "green",
                                                    "blue",
                                                    "black",
                                                    "purple"
                                                ]
                                            ],
                                            id="color",
                                            style={"color":"black"} # play nice (ish) with dark themes
                                        ),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col([html.H2("Lorems Ipsum"), html.P(id="paragraph")], width=8),
                    ]
                )
            ]
        ),
    ]
)


@app.callback(
    {"text": Output("paragraph", "children"), "style": Output("paragraph", "style")},
    Input("n_paragraphs", "value"),
    Input("color", "value"),
)
def generate_paragraphs(n, color):
    print(n, color)
    return dict(text=[html.P(p) for p in fake.paragraphs(nb=n)], style={"color": color})


#app.run(debug=False)


# ## Cytoscape
# 
# Now we can use the `dash-cytoscape` package to display our graph. Let's start with wireframe layout and then add in the functionality we need:

# In[ ]:


import pandas as pd
import networkx as nx

def create_graph(people_df: pd.DataFrame, attributes:list[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(((person["name"], person) for person in expanded_df.to_dict(orient="records")), person=True, type="person")
    
    for attribute in attributes:
        if attribute != "manager":
            add_nodes_from_attributes(G, attribute=attribute, flag=f"{attribute}_flag")
        add_edges_from_attributes(G, attribute=attribute)

    return G

def create_elements(attributes: list[str]=[]) -> list[dict]:
    """Generate a graph with connecting attributes and serialize it as cytoscape elements"""

    G = create_graph(people_df=people_df, attributes=attributes)
    elements = (
        nx.cytoscape_data(G)["elements"]["nodes"]
        + nx.cytoscape_data(G)["elements"]["edges"]
    )
    return elements


# In[147]:


def stylesheet_(focus:str=CEO["name"], theme:str="light", color:str=None, show_names:bool=False):
    dark = theme == "dark"
    return [
    {
        "selector": "node",
        "style": {
            "font-size": 50,
            "color": "lightgrey" if dark else "darkgrey"
        }
    },
    {
        "selector": "edge",
        "style": {
            "line-color": "lightgrey" if dark else "darkgrey",
            "color": "lightgrey" if dark else "darkgrey",
            "curve-style": "bezier",
            "label": "data(label)",
            "width": 1,
            "opacity": 0.25,
            "font-size": 10,
            "text-rotation": "autorotate",
        },
    },    
    {"selector": "edge[?languages]", "style": {"label": "codes"}},
    {"selector": "edge[?tz]", "style": {"label": "lives in"}},
    {"selector": "edge[?team]", "style": {"label": "belongs to"}},
    {"selector": "edge[?apps]", "style": {"label": "uses"}},
    {"selector": "edge[?manager]", "style": {"label": "manages", "source-arrow-shape":"triangle"}},     
    {
        "selector": "node[?person]",
        "style": {
            "label": "data(name)" if show_names else "",
            "background-color": "lightgreen" if dark else "green",
            "width": 25,
            "height": 25,            
            "font-size": 16
        },
    },
    {
        "selector": "node[?language_flag]",
        "style": {
            "label": "data(name)",
            "background-color": "white" if dark else "black",
            "width": 5,
            "height": 5,
        },
    },
    {
        "selector": "node[?tz_flag]",
        "style": {
            "label": "data(name)",
            "background-color": "white" if dark else "black",
            "width": 5,
            "height": 5,
        },
    },
    {
        "selector": f"node[id='{focus}']",
        "style":{
            "width": 25,
            "height": 25,
            "font-size": 20,
            "color":"skyblue",
            "background-color":"skyblue",
            "z-index":10            
        }
    },
    {
        "selector": "node[?app_flag]",
        "style": {"label": "data(name)", "width": 5, "height": 5},
    },
    {
        "selector": "node[?team_flag]",
        "style": {"label": "data(name)", "width": 5, "height": 5},
    },
    *node_color_stylesheet(attribute=color)
]

def node_color_stylesheet(attribute:str) -> dict:
    if not attribute:
        return []
    #colorscale = [f"var(--bs-{color})" for color in ["red","green","blue","pink","purple","yellow","indigo","cyan","orange","teal"]]
    colorscale = [color for color in ["red","green","blue","pink","purple","yellow","indigo","cyan","orange","teal"]]
    colors = dict(zip(expanded_df[attribute].unique(),colorscale))
    return [
        {"selector": f"node[{attribute}='{value}']", "style":{"background-color":color}}
        for value,color in colors.items()
    ]


# In[159]:


from dash import dash, html, dcc, Input, Output
import dash_cytoscape as cyto

cyto.load_extra_layouts()
dashboard = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

cyto_layout = {
    "name": "cose",
    "idealEdgeLength": 100,
    "nodeOverlap": 20,
    "refresh": 20,
    "fit": True,
    "padding": 30,
    "randomize": False,
    "componentSpacing": 100,
    "nodeRepulsion": 400000,
    "edgeElasticity": 100,
    "nestingFactor": 5,
    "gravity": 80,
    "numIter": 1000,
    "initialTemp": 200,
    "coolingFactor": 0.95,
    "minTemp": 1.0,
    "nodeDimensionsIncludeLabels": True,
}

dropdowns = [
    dbc.InputGroup(
        [
            dbc.InputGroupText("Attributes: "),
            dcc.Dropdown(
                value=None,
                options=[{"label": o, "value": o} for o in expanded_df.columns],
                id="attributes",
                placeholder="attrs as nodes",
                style={
                    "color": "black",
                    "min-width": "75px",
                },  # play nice (ish) with dark themes
                multi=True,
            ),
        ]
    ),
    dbc.InputGroup(
        [
            dbc.InputGroupText("Color: "),
            dcc.Dropdown(
                value=None,
                options=[{"label": o, "value": o} for o in expanded_df.columns],
                id="color",
                placeholder="attr as colors",
                style={
                    "color": "black",
                    "min-width": "75px",
                },  # play nice (ish) with dark themes
            ),
        ]
    ),
]


def layout():
    network = cyto.Cytoscape(
        id="network",
        layout=cyto_layout,
        responsive=True,
        style={"width": "100%", "height": "800px"},
        elements=create_elements(attributes=["team", "city"]),
        stylesheet=stylesheet_(theme="dark", color="country"),
    )
    return html.Div(
        [
            dbc.NavbarSimple(
                [
                    dbc.NavItem(
                        dbc.NavLink(
                            "PyData Seattle 2023",
                            href="https://pydata.org/seattle2023/schedule/",
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "Network Graph Tutorial",
                            href="https://lucasdurand.xyz/network-graph-tutorial",
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "Talk",
                            href="https://seattle2023.pydata.org/cfp/talk/83P9D7/",
                        )
                    ),
                ],
                brand="Peer Finder",
            ),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dropdowns,
                                width=4,
                                style={"background-color": "var(--bs-dark)"},
                            ),
                            dbc.Col(network, width=8),
                        ]
                    )
                ]
            ),
        ]
    )


dashboard.layout = layout

if __name__ == "__main__":
    dashboard.run(port=16900, debug=True, use_reloader=False)


# In[ ]:


app = dashboard.server
print("app loaded")


# ## Extras
# 
# * Change node size based on `rank` using a calculation
# * Customize the layout
# * Add a dropdown to choose how to colour nodes
#     * Pick an attribute to color on people nodes like plotly
#     * Color attribute nodes more statically
