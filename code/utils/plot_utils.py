import plotly
import plotly.graph_objects as go
import plotly.io as pio
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

def plot_cameras(Rs_pred, ts_pred, pts3D, Rs_gt, ts_gt, save_path):
    data = []
    data.append(get_3D_quiver_trace(ts_gt, Rs_gt[:, :3, 2], color='#86CE00', name='cam_gt'))
    data.append(get_3D_quiver_trace(ts_pred, Rs_pred[:, :3, 2], color='#C4451C', name='cam_learn'))
    data.append(get_3D_scater_trace(ts_gt.T, color='#86CE00', name='Ground truth cameras', size=2))
    data.append(get_3D_scater_trace(ts_pred.T, color='#C4451C', name='Predicted cameras', size=2))
    data.append(get_3D_scater_trace(pts3D, '#3366CC', '3D points', size=0.5))

    fig = go.Figure(data=data)
    fig.update_layout(showlegend=False)

    #fig.update_layout(
    #legend=dict(
    #    font=dict(family="Times New Roman, serif", size=38, color='black'),       # change the font size of legend text
    #    itemsizing='constant',    # ensures marker size is consistent
    #    traceorder='normal',      # order of legend items
    #)
#)
    camera = dict(
    eye=dict(x=0.42372829309514876,
    y=-0.363861827371949,
    z=-1.5813549717355537),
    center=dict(x=0.10168853835847104,
    y=-0.08282214300009132,
    z=-0.2530434569050146),
    up=dict(x=-0.007314272349394607,
    y=-0.9786736093325967,
    z=0.2052911781248936)
)
    fig.update_layout(scene_camera=camera)
    

    fig.update_scenes(
    xaxis=dict(showbackground=True, showgrid=True, showticklabels=False, title_text='', visible=False),
    yaxis=dict(showbackground=True, showgrid=True, showticklabels=False, title_text='', visible=False),
    zaxis=dict(showbackground=True, showgrid=True, showticklabels=False, title_text='', visible=False),
)
    fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0)  # left, right, top, bottom
)
    pio.write_image(fig, "yueh.svg", engine="kaleido", scale=3)
    plotly.offline.plot(fig, filename=save_path, auto_open=False)

    return save_path


def get_3D_quiver_trace(points, directions, color='#bd1540', name='', cam_size=1):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=directions[:, 0],
        v=directions[:, 1],
        w=directions[:, 2],
        sizemode='absolute',
        sizeref=cam_size,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_3D_scater_trace(points, color, name,size=0.5):
    assert points.shape[0] == 3, "3d plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d plot input points are not correctely shaped "

    trace = go.Scatter3d(
        name=name,
        x=points[0, :],
        y=points[1, :],
        z=points[2, :],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
        )
    )

    return trace