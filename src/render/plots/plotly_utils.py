def force_ddmmyyyy_axis(fig):
    try:
        fig.update_xaxes(
            tickformat="%d/%m/%Y",
            hoverformat="%d/%m/%Y",
            tickangle=0,
            automargin=True,
        )
    except Exception:
        pass
    return fig
