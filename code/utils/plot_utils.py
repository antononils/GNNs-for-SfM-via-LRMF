import plotly.graph_objects as go

def plotly_3d_points(X_final, save_path=None):
    pts = X_final[:, :3].detach().cpu().numpy()

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:,0],
                y=pts[:,1],
                z=pts[:,2],
                mode="markers",
                marker=dict(size=1, opacity=0.8)
            )
        ]
    )

    fig.update_layout(
        title="Interactive 3D Point Cloud Viewer",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=900,
        height=700,
    )

    if save_path is not None:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
