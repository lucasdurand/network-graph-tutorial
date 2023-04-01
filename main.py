# %% [markdown]
# # How Are People Connected?

# %% [markdown]
# ## The Data
#
# Just getting the data was difficult!
#
# * PIA concerns
# * consolidating multiple sources
# * highly available?
# * who owns this now?

# %% [markdown]
# For here let's use some sample data

# %%
import numpy as np
import pandas as pd

people = samplepeople = pd.read_csv(
    "https://raw.githubusercontent.com/lawlesst/vivo-sample-data/master/data/csv/people.csv"
)
people.title = people.title.str.strip()
people.head()

# %% [markdown]
# And add a new column with a relationship to another person. Let's pretend that everyone reports up to someone else in their stream, ideally someone with a "higher" title, unless they're all at the top of their profession, and then there is something like a "Chair" that they report to.

# %%
people.loc[people.title.str.contains("Professor"), "stream"] = "Academia"
people.loc[people.title.str.contains("Curator"), "stream"] = "Curation"

# %%
stream_ranks = [
    ["Professor", "Assistant Professor", "Research Professor"],
    ["Curator", "Associate Curator"],
]

# %%
ranks = {title: rank for stream in stream_ranks for rank, title in enumerate(stream)}
ranks

# %%
people["rank"] = people.title.map(ranks)

# %% [markdown]
# ### Generate Supervisory Org


# %%
def naivereportsto(row, df):
    supervisor = (
        df[(df.index < row.name)]
        .query(f"""rank <= {row["rank"]}-1""")
        .tail(1)
        .person_ID
    )
    supervisor = supervisor.item() if not supervisor.empty else None
    peer = (
        df[(df.index < row.name)].query(f"""rank  == {row["rank"]}""").head(1).person_ID
    )
    peer = peer.item() if not peer.empty else None
    return supervisor or peer or row.person_ID


# %%
def reportsto(df):
    return df.assign(manager=df.apply(naivereportsto, df=df, axis=1))


# %%
def supervisors(df):
    df["peoplemanager"] = df.apply(naivereportsto, df=df, axis=1)
    df = df.groupby("stream").apply(reportsto).reset_index(drop=True)
    return df


# %%
people = people.pipe(supervisors)
people.head(5)

# %% [markdown]
# Now let's turn this into a timeseries of data by semi-randomly giving people awards (from other people) over time

# %% [markdown]
# And let's add some more information about the individuals by giving everyone a list of subjects of speciality

# %%
subjects = ["Art", "English", "Science", "Medieval History", "Sports Cars"]

people["subjects"] = list(np.random.choice(subjects, size=(len(people), 2)))

# %%
people.head(5)

# %% [markdown]
# ## Build The Network


# %%
def elements(people):
    depth = people["rank"].max()
    size = 30
    return [
        {
            "data": {
                "id": person.get("person_ID"),
                "label": person.get("last"),
                "size": size * (depth - person.get("rank")),
                **person,
            },
        }
        for person in people.to_dict(orient="records")
    ]


# %%
def edges(people):
    manages = [
        {"data": {"label": "Manages", **edge}}
        for edge in people[["person_ID", "manager"]]
        .rename(columns={"person_ID": "source", "manager": "target"})
        .to_dict(orient="records")
        if edge.get("source") != edge.get("target")
    ]
    return manages


# %% [markdown]
# ## Dash

# %%
from dash import dash, html, dcc, Input, Output

# from jupyter_dash import JupyterDash
import dash_cytoscape as cyto

# JupyterDash.infer_jupyter_proxy_config()
cyto.load_extra_layouts()
dashboard = dash.Dash(__name__)
# dashboard = JupyterDash(__name__)

# %%
stylesheet = [
    # Group selectors
    {
        "selector": "node",
        "style": {
            "content": "data(label)",
            "width": "data(size)",
            "height": "data(size)",
        },
    },
    {"selector": """[stream = "Academia"]""", "style": {"background-color": "blue"}},
    {"selector": """[stream = "Curation"]""", "style": {"background-color": "green"}},
    # Edge selectors
    {
        "selector": "edge",
        "style": {
            "content": "data(label)",
            "curve-style": "bezier",
            "line-color": "gray",
            "source-arrow-shape": "triangle",
            "font-size": "10px",
        },
    },
]


# %%
def radial_circles(colours):
    stylesheet = {}
    return stylesheet


# %%
lyt = "cose-bilkent"
# lyt = "spread"
# lyt = "klay"


# %%
def layout():
    network = cyto.Cytoscape(
        id="network",
        layout={"name": lyt},
        style={"width": "100%", "height": "800px"},
        elements=elements(people) + edges(people),
        stylesheet=stylesheet,
    )
    return html.Div([html.H1("The University"), network])


dashboard.layout = layout

# %%
if __name__ == "__main__":
    dashboard.run_server(port=16900, debug=True, use_reloader=False)
else:
    app = dashboard.server

# %%
