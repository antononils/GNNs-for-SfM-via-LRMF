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
    data.append(get_3D_scater_trace(ts_gt.T, color='#86CE00', name='Ground truth camera', size=2))
    data.append(get_3D_scater_trace(ts_pred.T, color='#C4451C', name='Predicted camera', size=2))
    data.append(get_3D_scater_trace(pts3D, '#3366CC', '3D points', size=0.5))

    fig = go.Figure(data=data)
    fig.update_layout(
    legend=dict(
        font=dict(size=16),       # change the font size of legend text
        itemsizing='constant',    # ensures marker size is consistent
        traceorder='normal',      # order of legend items
    )
)
    camera = dict(
    eye=dict(x=0.8532378372509384, y=-0.5567518310038587, z=-0.5177441886434777),
    center=dict(x=-0.0050950933959899355, y=-0.000965791152714751, z=-0.09522949972226767),
    up=dict(x=-0.28740543144175656, y=-0.820062341570065, z=0.49486955242407066)
)
    fig.update_layout(scene_camera=camera)
    

    fig.update_scenes(
    xaxis=dict(showbackground=True, showgrid=True, showticklabels=False, title_text='', visible=False),
    yaxis=dict(showbackground=True, showgrid=True, showticklabels=False, title_text='', visible=False),
    zaxis=dict(showbackground=True, showgrid=True, showticklabels=False, title_text='', visible=False),
)
    pio.write_image(fig, "output2.svg", engine="kaleido", scale=3)
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