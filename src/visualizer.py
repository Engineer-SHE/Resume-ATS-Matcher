import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        self.color_scheme = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'info': '#3498DB'
        }

    def create_match_gauge(self, score: float) -> go.Figure:
        color = self._get_score_color(score)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={'text': "Overall Match Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 70, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#FFE5E5'},
                    {'range': [50, 70], 'color': '#FFF4E5'},
                    {'range': [70, 85], 'color': '#E5F5FF'},
                    {'range': [85, 100], 'color': '#E5FFE5'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(
            height=300,
            font={'size': 14},
            margin={'l': 20, 'r': 20, 't': 40, 'b': 20}
        )

        return fig

    def create_section_scores_chart(self, section_scores: Dict[str, float]) -> go.Figure:
        sections = list(section_scores.keys())
        scores = [score * 100 for score in section_scores.values()]
        colors = [self._get_score_color(score/100) for score in scores]

        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=sections,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{score:.1f}%' for score in scores],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title="Section-wise Match Scores",
            xaxis_title="Match Percentage",
            yaxis_title="Resume Section",
            height=400,
            xaxis=dict(range=[0, 105]),
            margin={'l': 100, 'r': 50, 't': 50, 'b': 50}
        )

        return fig

    def create_keyword_coverage_chart(self, gap_analysis: Dict[str, Any]) -> go.Figure:
        labels = ['Matched Keywords', 'Missing Keywords', 'Extra Keywords']
        values = [
            len(gap_analysis['matched_keywords']),
            len(gap_analysis['missing_keywords']),
            len(gap_analysis['extra_keywords'])
        ]
        colors = [self.color_scheme['success'], self.color_scheme['danger'], self.color_scheme['warning']]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=colors),
            textinfo='label+percent+value'
        )])

        fig.update_layout(
            title="Keyword Coverage Analysis",
            height=400,
            margin={'l': 20, 'r': 20, 't': 50, 'b': 20}
        )

        return fig

    def create_before_after_comparison(
        self,
        metrics_before: Dict[str, Any],
        metrics_after: Dict[str, Any]
    ) -> go.Figure:
        metrics = ['Keyword Coverage', 'Action Verbs', 'Matched Keywords']
        before_values = [
            metrics_before['keyword_coverage'] * 100,
            metrics_before['action_verbs_count'],
            metrics_before['matched_keywords']
        ]
        after_values = [
            metrics_after['keyword_coverage'] * 100,
            metrics_after['action_verbs_count'],
            metrics_after['matched_keywords']
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Before Optimization',
            x=metrics,
            y=before_values,
            marker_color=self.color_scheme['secondary'],
            text=[f'{v:.1f}' if i == 0 else f'{int(v)}' for i, v in enumerate(before_values)],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='After Optimization',
            x=metrics,
            y=after_values,
            marker_color=self.color_scheme['primary'],
            text=[f'{v:.1f}' if i == 0 else f'{int(v)}' for i, v in enumerate(after_values)],
            textposition='outside'
        ))

        fig.update_layout(
            title="ATS Optimization Impact",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            height=400,
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50}
        )

        return fig

    def create_keyword_heatmap(self, keyword_density: Dict[str, float]) -> go.Figure:
        if not keyword_density:
            return go.Figure()

        sorted_keywords = sorted(keyword_density.items(), key=lambda x: x[1], reverse=True)[:20]
        keywords = [kw[0] for kw in sorted_keywords]
        densities = [kw[1] for kw in sorted_keywords]

        n_cols = 4
        n_rows = (len(keywords) + n_cols - 1) // n_cols

        matrix = []
        idx = 0
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                if idx < len(densities):
                    row.append(densities[idx])
                    idx += 1
                else:
                    row.append(0)
            matrix.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Density %")
        ))

        annotations = []
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx < len(keywords):
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{keywords[idx]}<br>{densities[idx]:.1f}%",
                            showarrow=False,
                            font=dict(size=10)
                        )
                    )
                    idx += 1

        fig.update_layout(
            title="Keyword Density Heatmap",
            height=400,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            annotations=annotations,
            margin={'l': 20, 'r': 20, 't': 50, 'b': 20}
        )

        return fig

    def create_skills_radar_chart(self, skills_analysis: Dict[str, List[str]]) -> go.Figure:
        categories = ['Technical Skills', 'Soft Skills', 'Action Verbs', 'Domain Knowledge']
        values = [
            len(skills_analysis.get('technical', [])),
            len(skills_analysis.get('soft', [])),
            len(skills_analysis.get('action_verbs', [])),
            len(skills_analysis.get('domain', []))
        ]

        max_val = max(values) if values else 10
        normalized_values = [(v / max_val) * 100 if max_val > 0 else 0 for v in values]

        fig = go.Figure(data=go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            marker=dict(color=self.color_scheme['primary'])
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            title="Skills Distribution",
            height=400
        )

        return fig

    def create_action_verbs_chart(self, action_verb_analysis: Dict[str, Any]) -> go.Figure:
        if not action_verb_analysis.get('verb_counts'):
            return go.Figure()

        verb_counts = action_verb_analysis['verb_counts']
        sorted_verbs = sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        verbs = [v[0] for v in sorted_verbs]
        counts = [v[1] for v in sorted_verbs]

        fig = go.Figure(data=[
            go.Bar(
                x=counts,
                y=verbs,
                orientation='h',
                marker=dict(color=self.color_scheme['info']),
                text=counts,
                textposition='outside'
            )
        ])

        fig.update_layout(
            title="Top Action Verbs Used",
            xaxis_title="Frequency",
            yaxis_title="Action Verb",
            height=400,
            margin={'l': 100, 'r': 50, 't': 50, 'b': 50}
        )

        return fig

    def _get_score_color(self, score: float) -> str:
        if score >= 0.85:
            return self.color_scheme['success']
        elif score >= 0.70:
            return self.color_scheme['info']
        elif score >= 0.50:
            return self.color_scheme['warning']
        else:
            return self.color_scheme['danger']

    def generate_report_plots(self, analysis_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        plots = {}

        if 'overall_match' in analysis_results:
            plots['match_gauge'] = self.create_match_gauge(analysis_results['overall_match'])

        if 'section_scores' in analysis_results:
            plots['section_scores'] = self.create_section_scores_chart(analysis_results['section_scores'])

        if 'gap_analysis' in analysis_results:
            plots['keyword_coverage'] = self.create_keyword_coverage_chart(analysis_results['gap_analysis'])

        if 'metrics_before' in analysis_results and 'metrics_after' in analysis_results:
            plots['before_after'] = self.create_before_after_comparison(
                analysis_results['metrics_before'],
                analysis_results['metrics_after']
            )

        return plots