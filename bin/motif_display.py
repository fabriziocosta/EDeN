#!/usr/local/bin/python
import os
import sys
import re
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Graphics import GenomeDiagram
from reportlab.lib.units import cm
from reportlab.lib import colors
from Bio import SeqIO
import colorsys


def color_code(index, index_size):
    return colors.Color(*colorsys.hsv_to_rgb(index / float(index_size), 0.7, 0.8), alpha=0.5)

inFasta = str(sys.argv[1])
outDir = str(sys.argv[2])
motives_dir = os.path.join(outDir, "motives_dir")
html_files = os.path.join(motives_dir, "html_files")
if not os.path.exists(html_files):
    os.makedirs(html_files)

motifCoverageMap = {}
motifMap = {}
motifs_FH = open(os.path.join(outDir, "motifs.txt"))
for line in motifs_FH:
    matchObj = re.match(r'Cluster: ([0-9]+) #([0-9]+) \((.*)\)', line)
    if matchObj:
        motif_id = int(matchObj.group(1))
        no_of_hits = matchObj.group(2)
        seq_coverage = matchObj.group(3)
        motifCoverageMap[motif_id] = str(float(seq_coverage) * 100) + "%"
    if motif_id in motifMap:
        motifMap[motif_id] += line
    else:
        motifMap[motif_id] = line
motifs_FH.close()

handle = open(inFasta, "rU")
seqLenMap = {}
seqCount = 0
longest = 0
for record in SeqIO.parse(handle, "fasta"):
    seqId = record.id
    seqLen = len(record)
    seqLenMap[seqId] = seqLen
    longest = max(longest, seqLen)
    seqCount += 1
handle.close()

colorMap = [color_code(i, len(motifMap))for i in range(len(motifMap))]

seqColor = colors.grey.clone(alpha=0.2)

posBED_FH = open(os.path.join(outDir, "sequences_cluster_match_position.bed"), "r")
gdd = GenomeDiagram.Diagram()

prevSeqId = ""
trackId = 0
for line in posBED_FH:
    f = line.rstrip('\n').split('\t')
    seqId = f[0]
    seqLen = int(seqLenMap[seqId])
    padLen = max(0, int((longest - seqLen) / 2))
    start = int(f[1]) + padLen
    end = int(f[2]) + padLen
    cluster_id = int(f[3])
    if prevSeqId != seqId:
        gd_track = gdd.new_track(2 * trackId,
                                 greytrack=True,
                                 start=0,
                                 end=longest,
                                 name=seqId,
                                 height=1,
                                 greytrack_labels=1,
                                 greytrack_fontcolor=colors.black,
                                 greytrack_fontsize=round(seqCount / 2, 0),
                                 scale_color=colors.Color(1, 1, 1, 0))
        seqFeature = SeqFeature(FeatureLocation(padLen, seqLen + padLen), strand=None)
        seq_features = gd_track.new_set()
        seq_features.add_feature(seqFeature, color=seqColor)
        trackId += 1
    motif_features = gd_track.new_set()
    motifFeature = SeqFeature(FeatureLocation(start, end), strand=None)
    motif_features.add_feature(motifFeature,
                               color=colorMap[cluster_id],
                               label_position="middle",
                               label_angle=0,
                               label_size=round(seqCount / 2, 0),
                               name=str(cluster_id),
                               label=True)
    prevSeqId = seqId

gdd.draw(format='linear',
         pagesize=(longest * cm, round(seqCount * 2, 0) * cm),
         orientation='portrait',
         fragments=1,
         start=0,
         end=longest,
         y=0.0001)
gdd.write(os.path.join(outDir, "motives_overview.png"), "png")
posBED_FH.close()


index_HTML = open(os.path.join(outDir, "motives.html"), "w")
index_HTML.write("<div style=\"width:1200px\">")

motifCounter = 0
for f in os.listdir(motives_dir):
    if re.match('.*motif_[0-9]+.fa', f):
        motifCounter += 1

for f in range(0, motifCounter):
    fasta = os.path.join(motives_dir, "motif_" + str(f) + ".fa")
    muscleOutFa = fasta.replace(".fa", ".aligned.fa")
    muscleOutAln = fasta.replace(".fa", ".aln")
    weblogoOut = fasta.replace(".fa", ".png")
    os.system("muscle -in " + fasta + " -out " + muscleOutFa)
    os.system("muscle -in " + fasta + " -clw -out " + muscleOutAln)
    os.system("weblogo -f " + muscleOutFa + " -o " + weblogoOut +
              " -D fasta -F png_print -A rna -s large -c classic")
    index_HTML.write("<div align=\"center\" style= \"width:200px;height:150px;float:left;margin:20px 20px; \
    border: 0px solid #8AC007\"><a href=\"" + os.path.join("motives_dir", "html_files", "motif_" + str(f) +
                                                           ".html") + "\"><img style=\"width:100%;" +
                     "height:125px;border-bottom:0px solid #CCCCCC\"  src=\"" + "motives_dir/motif_" +
                     str(f) + ".png" + "\" alt=\"WEBLOGO\"></a><span>" + "motif_" + str(f) + " (" +
                     str(motifCoverageMap[f]) + ")</span></div>")

    current_HTML = open(os.path.join(html_files, "motif_" + str(f) + ".html"), "w")
    current_HTML.write("<div style=\"width:900px\">")
    current_HTML.write("<div style=\"float:right; width: 100%\"><img src=\"" + "../motif_" +
                       str(f) + ".png" + "\" alt=\"WEBLOGO\" style=\"width:100%\"></div>")
    current_HTML.write("<div style=\"clear:both\"></div><hr><hr>")
    current_HTML.write("<div align=\"center\" style=\"float:left;border: 1px solid #8AC007;width:270px; \
    height:500px\"><span style =\"display:block;margin-bottom:10px\">Sequences</span><object style=" +
                       "\"height:100%\" type=\"text/plain\" data=\"" +
                       "../motif_" + str(f) + ".fa" + "\"></object></div>")
    current_HTML.write("<div align=\"center\" style=\"float:left;border: 1px solid #8AC007;width:270px; " +
                       "height:500px; margin-left:10px\"><span style =\"display:block;margin-bottom:10px\">" +
                       "Alignment</span><object style=\"height:100%\"type=\"text/plain\" data=\"" +
                       "../motif_" + str(f) + ".aln" + "\"></object></div>")
    current_HTML.write("<div align=\"center\" style=\"float:left;border: 1px solid #8AC007; width:270px; " +
                       "height:500px; margin-left:10px\"><span style =\"display:block;margin-bottom:10px\">" +
                       "Motif Details</span><div align=\"left\" style=\"margin:8px\">" +
                       motifMap[f].replace('\n', "<br>") + "</div></div>")
    current_HTML.close()
index_HTML.write("</div>")
index_HTML.write(
    "<img style=\"float:left;clear:both;width:1200px;height:auto;border-bottom:1px solid #CCCCCC\"  " +
    "src=\"motives_overview.png\" alt=\"MOTIVES\">")

index_HTML.close()
