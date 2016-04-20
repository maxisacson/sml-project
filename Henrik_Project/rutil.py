import ROOT
from uuid import uuid4

def init_atlas_style():
    ROOT.gROOT.Reset()
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(1)
    ROOT.gStyle.SetPadLeftMargin(0.15)
    ROOT.gStyle.SetPadBottomMargin(0.15)
    #ROOT.gStyle.SetTitleXOffset(1.4)
    #ROOT.gStyle.SetTitleYOffset(1.4)

    ROOT.gStyle.SetTextFont(42)
    ROOT.gStyle.SetTextSize(0.05)

    font = 43
    font_size = 18

    ROOT.gStyle.SetLabelFont(font, 'x')
    ROOT.gStyle.SetTitleFont(font, 'x')
    ROOT.gStyle.SetLabelFont(font, 'y')
    ROOT.gStyle.SetTitleFont(font, 'y')
    ROOT.gStyle.SetLabelFont(font, 'z')
    ROOT.gStyle.SetTitleFont(font, 'z')

    ROOT.gStyle.SetLabelSize(font_size, 'x')
    ROOT.gStyle.SetTitleSize(font_size, 'x')
    ROOT.gStyle.SetLabelSize(font_size, 'y')
    ROOT.gStyle.SetTitleSize(font_size, 'y')
    ROOT.gStyle.SetLabelSize(font_size, 'z')
    ROOT.gStyle.SetTitleSize(font_size, 'z')

    ROOT.gStyle.SetTitleOffset(2.0, 'Y')
    ROOT.gStyle.SetTitleOffset(1.5, 'X')

    #ROOT.gStyle.SetErrorX(0.0001)

def show_hists(hists, title, file_name):
    colors = [ROOT.TColor.kRed, ROOT.TColor.kBlue, ROOT.TColor.kGreen,
            ROOT.TColor.kCyan, ROOT.TColor.kMagenta, ROOT.TColor.kYellow]
    colors = colors + [ROOT.TColor.kBlack]*(len(hists) - len(colors))

    canvas = ROOT.TCanvas(uuid4().hex, title, 0, 0, 800, 600)
    legend = ROOT.TLegend(0.70, 0.80, 0.90, 0.90)
    same = ''
    draw_legend = False

    for h, c in zip(hists, colors):
        h.SetTitle(title)
        h.SetMarkerColor(c)
        h.SetLineColor(c)
        if isinstance(h, ROOT.TH2):
            h.Draw('COLZ')
            draw_legend = False
        elif isinstance(h, ROOT.TH1):
            h.Draw(same + ' hist E')
            same = 'same'
            legend.AddEntry(h, h.GetName(), 'l')
            draw_legend = True
        else:
            h.Draw(same + 'ep')
            same = 'same'
            legend.AddEntry(h, h.GetName(), 'l')
            draw_legend = True
    if draw_legend:
        legend.Draw()
    canvas.SaveAs(file_name)
    return (canvas, legend, hists)
