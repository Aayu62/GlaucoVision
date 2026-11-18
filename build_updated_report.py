"""
Build Updated GlaucoVision Report PDF
Generates an 8-page IEEE conference/transactions style paper using ReportLab.
Fixes:
1. Missing mathematical symbols: replaces '□' with '∈' and '⊙'.
2. Cleans up subscript formatting and equations (w1, w2, w3, Fch, Fsp).
3. Updates Section VIII (Limitations) to reflect that requirements.txt is provided.
4. Adds live Hugging Face Spaces (ZeroGPU) and GitHub deployment details.
5. Updates Appendix A repository layout with requirements.txt and app.py.
6. Embeds Figure 1 (Architecture) and Figure 2 (Confusion Matrix).
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

# Register TrueType Fonts for Unicode and Math support
pdfmetrics.registerFont(TTFont('SegoeUI', 'C:/Windows/Fonts/segoeui.ttf'))
pdfmetrics.registerFont(TTFont('SegoeUI-Bold', 'C:/Windows/Fonts/segoeuib.ttf'))
pdfmetrics.registerFont(TTFont('SegoeUI-Italic', 'C:/Windows/Fonts/segoeuii.ttf'))
pdfmetrics.registerFont(TTFont('SegoeUI-BoldItalic', 'C:/Windows/Fonts/segoeuiz.ttf'))
pdfmetrics.registerFont(TTFont('SegoeUISymbol', 'C:/Windows/Fonts/seguisym.ttf'))
pdfmetrics.registerFontFamily('SegoeUI', normal='SegoeUI', bold='SegoeUI-Bold', italic='SegoeUI-Italic', boldItalic='SegoeUI-BoldItalic')

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y'
    along with running headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('SegoeUI', 6.8)
        self.setFillColor(colors.HexColor('#555555'))
        
        # Header (pages 1 to N)
        self.drawString(
            56.69, 841.89 - 28,
            "GlaucoVision: Multistage Glaucoma Classification Using a Cascaded Attention Network (CA-Net) with Explainable AI and Web Deployment"
        )
        
        # Footer
        self.drawString(
            56.69, 24,
            "GlaucoVision: Multistage Glaucoma Classification Using a Cascaded Attention Network (CA-Net) with Explainable AI and Web Deployment"
        )
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(595.28 - 56.69, 24, page_str)
        self.restoreState()

def build_pdf(filename="GlaucoVision_Report_Updated.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=56.69,
        rightMargin=56.69,
        topMargin=36,
        bottomMargin=36
    )

    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='SegoeUI-Bold',
        fontSize=20,
        leading=24,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#111827'),
        spaceAfter=10
    )
    
    author_style = ParagraphStyle(
        'AuthorLine',
        parent=styles['Normal'],
        fontName='SegoeUI-Bold',
        fontSize=9.2,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=2
    )

    affil_style = ParagraphStyle(
        'AffilLine',
        parent=styles['Normal'],
        fontName='SegoeUI',
        fontSize=9.0,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#374151'),
        spaceAfter=2
    )

    corresp_style = ParagraphStyle(
        'CorrespLine',
        parent=styles['Normal'],
        fontName='SegoeUI-Italic',
        fontSize=8.8,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=10
    )

    sec_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='SegoeUI-Bold',
        fontSize=13,
        leading=17,
        spaceBefore=11,
        spaceAfter=5,
        textColor=colors.HexColor('#0f172a'),
        keepWithNext=True
    )

    subsec_style = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Normal'],
        fontName='SegoeUI-Bold',
        fontSize=9.6,
        leading=13.5,
        spaceBefore=7,
        spaceAfter=3,
        textColor=colors.HexColor('#1e293b'),
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='SegoeUI',
        fontSize=9.2,
        leading=13.0,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        textColor=colors.HexColor('#1e293b')
    )

    abstract_style = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='SegoeUI',
        fontSize=9.0,
        leading=12.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        textColor=colors.HexColor('#1e293b')
    )

    caption_style = ParagraphStyle(
        'FigureCaption',
        parent=styles['Normal'],
        fontName='SegoeUI-Italic',
        fontSize=8.0,
        leading=11,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=7,
        textColor=colors.HexColor('#334155')
    )

    tbl_caption_style = ParagraphStyle(
        'TableCaption',
        parent=styles['Normal'],
        fontName='SegoeUI-Bold',
        fontSize=8.0,
        leading=11,
        alignment=TA_LEFT,
        spaceBefore=6,
        spaceAfter=3,
        textColor=colors.HexColor('#0f172a'),
        keepWithNext=True
    )

    code_style = ParagraphStyle(
        'CodeListing',
        parent=styles['Normal'],
        fontName='SegoeUI',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor('#1e293b')
    )

    ref_style = ParagraphStyle(
        'ReferenceItem',
        parent=styles['Normal'],
        fontName='SegoeUI',
        fontSize=8.3,
        leading=11.5,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leftIndent=18,
        firstLineIndent=-18,
        textColor=colors.HexColor('#1e293b')
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("GlaucoVision: Multistage Glaucoma Classification Using a Cascaded Attention Network (CA-Net) with Explainable AI and Web Deployment", title_style))
    story.append(Paragraph("Jiya Gupta (Roll No. 2305380), Ayush Kumar (Roll No. 2305371), Anurima Ghosh (Roll No. 2305360)", author_style))
    story.append(Paragraph("School of Computer Engineering, KIIT University, Bhubaneswar, Odisha, India", affil_style))
    story.append(Paragraph("Guide: Prof. Parthapratim Sarangi", affil_style))
    story.append(Paragraph("Corresponding author: anurimrghosh16@gmail.com", corresp_style))

    story.append(Paragraph("<b>Abstract</b>", ParagraphStyle('AbsH', parent=subsec_style, fontSize=9.5, spaceBefore=4, spaceAfter=2)))
    story.append(Paragraph(
        "Glaucoma is an irreversible optic neuropathy, and early-stage disease is hard to separate from advanced disease "
        "on fundus photographs, so most automated systems stop at a binary normal-versus-glaucoma decision. This paper "
        "presents GlaucoVision, an end-to-end system for three-class glaucoma staging (Normal, Early-stage, "
        "Advanced-stage) built on the Cascaded Attention Network (CA-Net). CA-Net places a Cascaded Attention Module, a "
        "Triplet Channel Attention Block (TCAB) followed by a Spatial Attention Block (SAB), on an ImageNet-pretrained "
        "DenseNet-121 backbone. We reconstruct the Large Multistage Glaucoma (LMG) dataset of 1,582 fundus images "
        "from Harvard Dataverse V1 (1,542 images) and RIM-ONE r1 (40 images), using stratified splits of 774 training, 333 "
        "validation and 475 test images. Under identical 50-epoch training, CA-Net reaches 77.05% test accuracy, 74.17% "
        "macro F1, 76.05% macro recall and a multi-class AUC of 0.9112, improving on the plain DenseNet-121 baseline "
        "(74.74% accuracy, 69.33% macro F1, 70.30% macro recall, AUC 0.8974) by 2.31, 4.84 and 5.75 percentage points "
        "in accuracy, F1 and recall. Advanced-stage recall is 0.88, whereas Early-stage remains the hardest class (F1 0.58). "
        "Grad-CAM and Grad-CAM++ heatmaps show that the model attends to the optic cup and neuroretinal rim. The "
        "trained model is deployed through a FastAPI backend and a browser dashboard that returns stage probabilities and "
        "heatmap overlays in real time.",
        abstract_style
    ))

    story.append(Paragraph(
        "<b>Index Terms</b>—Glaucoma, fundus image classification, attention mechanism, DenseNet-121, Grad-CAM, "
        "explainable AI, medical image analysis, FastAPI, ZeroGPU.",
        abstract_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("I. Introduction", sec_style))
    story.append(Paragraph(
        "Glaucoma is a chronic neurodegenerative eye disease in which retinal ganglion cells and optic nerve fibers are lost "
        "progressively, usually under raised intraocular pressure. It is the leading cause of irreversible blindness worldwide, "
        "and more than 110 million people are projected to be affected by 2040 [1]. Because peripheral vision fades without "
        "pain, it is often called the “silent thief of sight”; lost vision cannot be restored, but eye drops (for example "
        "prostaglandin analogues) or surgery can halt progression if treatment starts early.",
        body_style
    ))
    story.append(Paragraph(
        "The disease shows on color fundus photographs as structural change at the optic nerve head: the optic cup "
        "enlarges and the neuroretinal rim thins. The vertical cup-to-disc ratio (CDR) is about 0.2 to 0.4 in healthy eyes and "
        "rises past 0.6 to 0.8 as fibers degenerate. Reading such photographs manually is slow and varies between "
        "observers, and ophthalmologists are scarce in rural and low-resource clinics, where instrument-based perimetry is "
        "also often unavailable. This motivates automated screening.",
        body_style
    ))
    story.append(Paragraph(
        "Most deep learning systems perform binary detection [2], although early and advanced glaucoma need different "
        "treatment pathways. Staging is hard because early lesions are microscopic rim notches, the classes look alike, and "
        "roughly 85 to 90% of a fundus photograph is irrelevant background. Ahn et al. made an early attempt at multistage "
        "classification [3]. CA-Net [4] addresses these issues with cross-dimensional channel attention cascaded with spatial "
        "attention. This work reproduces and extends it. The contributions are:",
        body_style
    ))
    story.append(Paragraph(
        "1. Reconstruction of the 1,582-image LMG benchmark from Harvard Dataverse V1 and RIM-ONE r1 with stratified splits.",
        body_style
    ))

    # ==================== PAGE 2 ====================
    story.append(PageBreak())
    story.append(Paragraph(
        "2. A PyTorch implementation of the Cascaded Attention Module (TCAB and SAB, with learnable branch weights "
        "<i>w</i><sub>1</sub>, <i>w</i><sub>2</sub>, <i>w</i><sub>3</sub>) on DenseNet-121, with a no-attention option used for the baseline.",
        body_style
    ))
    story.append(Paragraph(
        "3. A like-for-like comparison with a baseline DenseNet-121 trained under the same 50-epoch protocol, reported "
        "with overall, per-class and confusion-matrix results.",
        body_style
    ))
    story.append(Paragraph(
        "4. Grad-CAM and Grad-CAM++ explanations that localize predictions to the optic nerve head.",
        body_style
    ))
    story.append(Paragraph(
        "5. A full-stack web decision-support tool built with FastAPI and a browser dashboard, alongside live cloud deployment on Hugging Face Spaces (ZeroGPU).",
        body_style
    ))

    story.append(Paragraph("II. Background and Related Work", sec_style))
    story.append(Paragraph(
        "<b>A. Classical and deep learning approaches.</b> Early screening systems segmented the optic disc and cup and "
        "computed clinical measures, or fed handcrafted texture features such as wavelet energy to classifiers like SVMs [5]; "
        "these depend on boundary contrast, are sensitive to vessel obscuration and generalize poorly across cameras. "
        "Deep CNNs learn features directly from pixels. DENet combined whole-image and disc-region streams for binary "
        "screening [2], Ahn et al. separated early from advanced glaucoma on Harvard Dataverse but struggled with "
        "inter-class similarity [3], and Parashar and Agrawal classified glaucoma stages with 2-D compact variational mode "
        "decomposition [6].",
        body_style
    ))
    story.append(Paragraph(
        "<b>B. Attention mechanisms.</b> Squeeze-and-Excitation networks weight channels only [7]. CBAM applies channel then "
        "spatial attention, but its spatial branch pools channels naively with max and average operations [8]. Triplet Attention "
        "captures cross-dimension interaction by permuting tensors, yet sums its branches with fixed weights and has no "
        "local channel attention [9]. CABNet and GC-Net apply category and global-class attention to retinal grading and "
        "glaucoma classification [10], [11]. CA-Net [4] combines 1D channel attention with channel-width and channel-height "
        "cross-attention in TCAB, learns the branch weights, and cascades the result into a spatial attention block.",
        body_style
    ))
    story.append(Paragraph(
        "<b>C. Explainability.</b> Grad-CAM weights activation maps by spatially averaged gradients of the class score [12]. "
        "Grad-CAM++ adds higher-order gradient terms, which localizes multiple or small lesions better [13]. Visual "
        "explanations are needed for clinical acceptance of any automated stage prediction.",
        body_style
    ))

    story.append(Paragraph("III. Dataset", sec_style))
    story.append(Paragraph(
        "<b>A. Source datasets.</b> Harvard Dataverse V1 holds fundus photographs taken at Kim’s Eye Hospital, Seoul, with an "
        "AFC-330 non-mydriatic camera and pre-cropped to 240 × 240 pixels around the optic nerve head. It contributes 786 "
        "Normal, 289 Early-stage and 467 Advanced-stage images (1,542 in total; 786 of 788 available normal files were "
        "used). RIM-ONE r1 [14] has 158 patient folders containing raw photographs (im*.bmp), multi-expert segmentation "
        "masks (-exp1 to -exp5) and gold-standard coordinate files; a filename filter (^im\\d+\\.bmp$) keeps only the raw "
        "photographs. From it we take 12 Early-stage and 14 Advanced-stage images and sample 14 Normal images from "
        "the 118 available, giving 40 images. The merged LMG dataset has 1,582 images (Table I).",
        body_style
    ))

    # TABLE I
    story.append(Paragraph("TABLE I. COMPOSITION OF THE LMG DATASET", tbl_caption_style))
    t1_data = [
        ["Source", "Normal", "Early-stage", "Advanced-stage", "Total"],
        ["Harvard Dataverse V1", "786", "289", "467", "1,542"],
        ["RIM-ONE r1", "14", "12", "14", "40"],
        ["LMG total", "800", "301", "481", "1,582"]
    ]
    t1 = Table(t1_data, colWidths=[160, 80, 80, 80, 81.9])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>B. Stratified partitioning.</b> Following the CA-Net protocol [4], the images are split 70:30 into training-plus-validation "
        "(1,107 images) and test (475 images) sets, and the first part is split 70:30 again into 774 training and 333 validation "
        "images (48.9%, 21.1% and 30.0% of the data). All splits are stratified by class label, so class proportions are "
        "preserved (Table II); the split membership is stored as CSV files so every run uses identical partitions.",
        body_style
    ))

    # ==================== PAGE 3 ====================
    story.append(PageBreak())
    story.append(Paragraph("TABLE II. STRATIFIED TRAIN, VALIDATION AND TEST SPLITS", tbl_caption_style))
    t2_data = [
        ["Partition", "Normal", "Early-stage", "Advanced-stage", "Total"],
        ["Train", "392 (50.6%)", "147 (19.0%)", "235 (30.4%)", "774"],
        ["Validation", "168 (50.5%)", "64 (19.2%)", "101 (30.3%)", "333"],
        ["Test", "240 (50.5%)", "90 (18.9%)", "145 (30.5%)", "475"],
        ["Total", "800", "301", "481", "1,582"]
    ]
    t2 = Table(t2_data, colWidths=[110, 95, 95, 100, 81.9])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>C. Preprocessing and augmentation.</b> Images are resized bilinearly to 224 × 224 pixels. Training augmentation, "
        "applied on the fly in the data loader, uses horizontal and vertical flips (p = 0.5 each), rotation within ±20° and affine "
        "scaling between 0.9 and 1.1. Pixel values are scaled to [0, 1] and then normalized with the ImageNet mean (0.485, "
        "0.456, 0.406) and standard deviation (0.229, 0.224, 0.225), giving a (3, 224, 224) tensor per image and a (16, 3, "
        "224, 224) mini-batch. Validation and test images are only resized and normalized.",
        body_style
    ))

    story.append(Paragraph("IV. Methodology", sec_style))
    story.append(Paragraph(
        "<b>A. Overall architecture.</b> CA-Net has three sequential stages: an ImageNet-pretrained DenseNet-121 backbone "
        "[15], the Cascaded Attention Module (TCAB then SAB), and a classifier head of global average pooling (GAP), a "
        "fully connected layer (1024 → 3) and softmax (Fig. 1). Dense connectivity reuses features and eases gradient flow. "
        "For a 224 × 224 × 3 input the backbone, followed by a ReLU, outputs <i>F</i><sub>in</sub> with <i>K</i> = 1024 channels on a 7 × 7 grid. "
        "The full model has about 7.09 million trainable parameters.",
        body_style
    ))

    # Figure 1: Architecture
    fig1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "figure1_architecture.png")
    if os.path.exists(fig1_path):
        img1 = Image(fig1_path, width=440, height=192)
        img1.hAlign = 'CENTER'
        story.append(img1)
        story.append(Paragraph("Fig. 1. CA-Net: DenseNet-121 backbone, Cascaded Attention Module (TCAB then SAB) and classifier head.", caption_style))

    # Table III: Tensor shapes
    story.append(Paragraph("TABLE III. TENSOR SHAPES THROUGH CA-NET (BATCH SIZE B = 16)", tbl_caption_style))
    t3_data = [
        ["Stage", "Output shape", "Operation"],
        ["Input image", "(B, 3, 224, 224)", "Resize, augment, normalize"],
        ["DenseNet-121 + ReLU", "(B, 1024, 7, 7)", "Pretrained feature extractor, F_in"],
        ["TCAB", "(B, 1024, 7, 7)", "Three attention branches fused by w_1, w_2, w_3 -> F_ch"],
        ["SAB", "(B, 1024, 7, 7)", "CCAP, 1×1 conv, LN, sigmoid mask, F_sp"],
        ["GAP", "(B, 1024)", "Spatial average per channel"],
        ["Linear + softmax", "(B, 3)", "Logits and class probabilities"]
    ]
    t3 = Table(t3_data, colWidths=[120, 110, 251.9])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>B. Triplet Channel Attention Block (TCAB).</b> TCAB models dependencies across channel, channel-width and "
        "channel-height dimensions with three parallel branches:",
        body_style
    ))
    story.append(Paragraph(
        "1. <i>Branch 1, 1D channel attention:</i> GAP over the spatial grid, a 1×1 convolution to <i>K/r</i> channels (reduction ratio <i>r</i> = "
        "16), layer normalization (LN), ReLU, a 1×1 convolution back to <i>K</i> channels, LN and a sigmoid, multiplied with <i>F</i><sub>in</sub> "
        "to give <i>F</i><sub>ch1</sub>.",
        body_style
    ))

    # ==================== PAGE 4 ====================
    story.append(PageBreak())
    story.append(Paragraph(
        "2. <i>Branch 2, channel-width cross attention:</i> permute the tensor to (B, H, K, W), apply cross-height average pooling "
        "(CHAP) over H to obtain a (B, 1, K, W) descriptor, then a 1×1 convolution, LN and sigmoid, multiply and permute "
        "back to give <i>F</i><sub>ch2</sub>.",
        body_style
    ))
    story.append(Paragraph(
        "3. <i>Branch 3, channel-height cross attention:</i> permute the tensor to (B, W, H, K), apply cross-width average pooling "
        "(CWAP) over W to obtain a (B, 1, H, K) descriptor, then a 1×1 convolution, LN and sigmoid, multiply and "
        "permute back to give <i>F</i><sub>ch3</sub>. This branch helps capture vertical cup elongation.",
        body_style
    ))
    story.append(Paragraph("The branch outputs are fused with three learnable scalar weights:", body_style))
    
    # Equation 1: TCAB fusion
    eq1_style = ParagraphStyle('Eq1', parent=styles['Normal'], fontName='SegoeUI-Bold', fontSize=10, alignment=TA_CENTER, spaceBefore=4, spaceAfter=4)
    story.append(Paragraph("<i>F</i><sub>ch</sub> = <i>w</i><sub>1</sub> · <i>F</i><sub>ch1</sub> + <i>w</i><sub>2</sub> · <i>F</i><sub>ch2</sub> + <i>w</i><sub>3</sub> · <i>F</i><sub>ch3</sub>", eq1_style))
    
    story.append(Paragraph(
        "The three weights are initialized at 1/3 and optimized end to end with the rest of the network by backpropagation, so "
        "the model learns how much each dimension matters. Their learned values are shown in the web interface.",
        body_style
    ))
    story.append(Paragraph(
        "<b>C. Spatial Attention Block (SAB).</b> SAB applies cross-channel average pooling (CCAP) over all <i>K</i> channels of <i>F</i><sub>ch</sub>, "
        "followed by a 1×1 convolution, LN and a sigmoid to produce a spatial mask <i>M</i><sub>sp</sub> of shape (B, 1, H, W). The mask "
        "emphasizes regions such as the optic cup and neuroretinal rim and suppresses background; it is multiplied with "
        "every channel of <i>F</i><sub>ch</sub>, yielding <b><i>F</i><sub>sp</sub> = <i>M</i><sub>sp</sub> <font name=\"SegoeUISymbol\">&#x2299;</font> <i>F</i><sub>ch</sub></b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>D. Cascaded Attention Module (CAM) and classifier.</b> CAM chains the two blocks, <i>F</i><sub>in</sub> → TCAB → <i>F</i><sub>ch</sub> → SAB → "
        "<i>F</i><sub>sp</sub>. The refined features are pooled by GAP into a 1024-dimensional vector and classified by a linear layer, <i>z</i> = "
        "GAP(<i>F</i><sub>sp</sub>)<i>W</i><sup><i>T</i></sup> + <i>b</i>, and a softmax gives class probabilities, "
        "<i>ŷ</i><sub><i>c</i></sub> = exp(<i>z</i><sub><i>c</i></sub>) / ∑<sub><i>j</i></sub> exp(<i>z</i><sub><i>j</i></sub>), for <b><i>c</i> <font name=\"SegoeUISymbol\">&#x2208;</font> {Normal, Early-stage, Advanced-stage}</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>E. Loss function.</b> Training minimizes the categorical cross-entropy, <i>L</i> = −(1/<i>B</i>) ∑<sub><i>b</i></sub> ∑<sub><i>c</i></sub> <i>y</i><sub><i>b,c</i></sub> log <i>ŷ</i><sub><i>b,c</i></sub>, averaged over the "
        "mini-batch. For an Early-stage image predicted as (0.24, 0.60, 0.16), for instance, the loss is −log 0.60 ≈ 0.51; a "
        "confident correct prediction gives a loss near zero and a confident wrong one a large penalty.",
        body_style
    ))
    story.append(Paragraph(
        "<b>F. Explainability.</b> The last feature map (the CAM output) and its gradients with respect to the target class score are "
        "captured with hooks. Grad-CAM averages the gradients to obtain channel weights α<sub><i>k</i></sub><sup><i>c</i></sup> = (1/<i>uv</i>) ∑<sub><i>i,j</i></sub> ∂<i>y</i><sup><i>c</i></sup>/∂<i>A</i><sub><i>ij</i></sub><sup><i>k</i></sup> and "
        "forms ReLU(∑<sub><i>k</i></sub> α<sub><i>k</i></sub><sup><i>c</i></sup> <i>A</i><sup><i>k</i></sup>), whereas Grad-CAM++ weights each spatial gradient using second- and third-order gradient "
        "terms for finer localization of small or multiple lesions. The normalized heatmap is upsampled to 224 × 224, "
        "colorized with the Jet colormap and alpha-blended over the original fundus image.",
        body_style
    ))

    story.append(Paragraph("V. System Implementation", sec_style))
    story.append(Paragraph(
        "<b>A. Software stack.</b> The system is written in Python 3.11 and uses PyTorch [16] and Torchvision for modelling, "
        "OpenCV and Pillow for image handling, Pandas, NumPy and scikit-learn for data management and metrics, "
        "Matplotlib for figures, and FastAPI served by Uvicorn for the backend. The frontend is a vanilla "
        "HTML5/CSS3/JavaScript single-page application without a build step.",
        body_style
    ))
    story.append(Paragraph(
        "<b>B. Backend & Deployment.</b> The model and explainer are loaded once at startup from the best checkpoint (about 86 MB) onto GPU "
        "if available, otherwise CPU, and set to evaluation mode. The system is deployed via two parallel channels: (1) a local FastAPI "
        "REST server exposing the endpoints in Table IV, and (2) a cloud-native Hugging Face Space utilizing NVIDIA ZeroGPU (Free Tier) "
        "with an interactive Gradio interface for global access without hardware costs.",
        body_style
    ))

    # TABLE IV
    story.append(Paragraph("TABLE IV. REST API ENDPOINTS", tbl_caption_style))
    t4_data = [
        ["Endpoint", "Method", "Function"],
        ["/api/status", "GET", "Model status, device, backbone and learned TCAB weights"],
        ["/api/predict", "POST", "Predicted stage, confidence, class probabilities, clinical description, latency"],
        ["/api/explain", "POST", "Original image, Grad-CAM and Grad-CAM++ overlays as Base64 data URLs"],
        ["/api/samples", "GET", "Benchmark sample images for each stage"]
    ]
    t4 = Table(t4_data, colWidths=[90, 60, 331.9])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t4)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>C. Frontend.</b> The interface provides drag-and-drop upload, preset benchmark samples, a diagnostic verdict banner, "
        "probability bars for the three classes (green, amber and red for Normal, Early and Advanced), a live display of the "
        "TCAB weights, and a multi-view explainability viewer (Grad-CAM++, Grad-CAM, side-by-side, original) with a "
        "heatmap opacity slider from 0% (raw fundus) to 100% (full thermal map). A notice states that the tool is a research "
        "prototype and does not replace evaluation by an ophthalmologist. Client and server communicate through JSON "
        "and Base64 payloads over the fetch API.",
        body_style
    ))

    # ==================== PAGE 5 ====================
    story.append(PageBreak())
    story.append(Paragraph(
        "<b>D. Inference workflow.</b> An uploaded image is decoded and converted to RGB, resized to 224 × 224, normalized "
        "and batched to shape (1, 3, 224, 224). The model produces logits and probabilities, the explainer back-propagates "
        "the target class score, and the server returns the verdict together with the heatmap overlays.",
        body_style
    ))

    story.append(Paragraph("VI. Experimental Setup", sec_style))
    story.append(Paragraph(
        "Both models were trained for 50 epochs on the 774 training images (49 mini-batches of 16 per epoch, the last "
        "holding 6 images) with the Adam optimizer and categorical cross-entropy loss. After each epoch the model is "
        "evaluated on the 333 validation images (accuracy, macro precision, recall, F1 and AUC), the ReduceLROnPlateau "
        "scheduler is stepped with the validation F1, and the checkpoint with the highest validation macro F1 is retained. "
        "Training and validation loss, accuracy and F1 are logged per epoch to JSON. A baseline DenseNet-121 without "
        "attention is trained under the same protocol for ablation. Training and evaluation ran on a 12th-generation Intel Core "
        "i7-1255U CPU (10 cores, 12 threads) with PyTorch, Torchvision, OpenCV and scikit-learn. Evaluation on the "
        "held-out test set reports accuracy, per-class precision, recall and F1, a 3 × 3 confusion matrix and one-vs-rest ROC "
        "curves with macro AUC. The settings are listed in Table V.",
        body_style
    ))

    # TABLE V
    story.append(Paragraph("TABLE V. TRAINING CONFIGURATION", tbl_caption_style))
    t5_data = [
        ["Setting", "Value"],
        ["Input size", "224 × 224 × 3, bilinear resize"],
        ["Mini-batch size", "16"],
        ["Epochs", "50"],
        ["Optimizer", "Adam (β_1 = 0.9, β_2 = 0.999, ε = 1e-8)"],
        ["Initial learning rate", "0.005"],
        ["Scheduler", "ReduceLROnPlateau on validation F1 (mode = max, factor = 0.5, patience = 5)"],
        ["Loss", "Categorical cross-entropy"],
        ["Checkpoint criterion", "Highest validation macro F1"],
        ["Backbone initialization", "ImageNet-pretrained DenseNet-121"]
    ]
    t5 = Table(t5_data, colWidths=[150, 331.9])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t5)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "With patience 5 and factor 0.5, the learning rate is halved (0.005, 0.0025, 0.00125, ...) whenever validation F1 fails "
        "to improve for five consecutive epochs, so that the network takes smaller steps as it converges.",
        body_style
    ))

    story.append(Paragraph("VII. Results and Discussion", sec_style))
    story.append(Paragraph(
        "<b>A. Ablation against the baseline.</b> Table VI compares the two models on the 475 held-out test images. CA-Net is "
        "better on every metric: +2.31 points of accuracy, +3.92 of macro precision, +5.75 of macro recall and +4.84 of macro "
        "F1, a 0.0138 higher multi-class AUC and a lower test loss. Recall rises most, which matters in screening because a "
        "missed case is an undiagnosed patient. The best validation accuracy during CA-Net training was 79.28%, about two "
        "points above test accuracy.",
        body_style
    ))

    # TABLE VI
    story.append(Paragraph("TABLE VI. TEST-SET PERFORMANCE, BASELINE VERSUS CA-NET (475 IMAGES)", tbl_caption_style))
    t6_data = [
        ["Metric", "DenseNet-121 (baseline)", "CA-Net (DenseNet-121 + CAM)", "Change"],
        ["Accuracy (%)", "74.74", "77.05", "+2.31"],
        ["Macro precision (%)", "69.61", "73.53", "+3.92"],
        ["Macro recall (%)", "70.30", "76.05", "+5.75"],
        ["Macro F1 (%)", "69.33", "74.17", "+4.84"],
        ["Multi-class AUC", "0.8974", "0.9112", "+0.0138"],
        ["Cross-entropy test loss", "0.5730", "0.5347", "−0.0383 (lower is better)"]
    ]
    t6 = Table(t6_data, colWidths=[130, 115, 135, 101.9])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t6)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>B. Per-class results.</b> Table VII and Fig. 2 break down the CA-Net predictions. Advanced-stage disease is found "
        "most reliably (recall 0.88; 127 of 145 correct, only 5 called Normal). Normal images are identified with high precision "
        "(0.91) but lower recall (0.75), and 42 of the 240 are labelled Early-stage. Early-stage is the weakest class: 59 of 90 "
        "are recovered (recall 0.66) but precision is only 0.52, because Normal images are the main source of false "
        "Early-stage calls (42 of the 114 Early-stage predictions). The accuracy of 77.05% is therefore moderate, which is "
        "typical of the difficult boundary between Normal and Early-stage glaucoma, while the AUC of 0.911 indicates good "
        "class separability.",
        body_style
    ))

    # ==================== PAGE 6 ====================
    story.append(PageBreak())
    story.append(Paragraph("TABLE VII. PER-CLASS REPORT FOR CA-NET ON THE TEST SET", tbl_caption_style))
    t7_data = [
        ["Stage", "Precision", "Recall", "F1-score", "Support"],
        ["Normal", "0.91", "0.75", "0.82", "240"],
        ["Early-stage", "0.52", "0.66", "0.58", "90"],
        ["Advanced-stage", "0.78", "0.88", "0.82", "145"],
        ["Macro average", "0.74", "0.76", "0.74", "475"],
        ["Weighted average", "0.80", "0.77", "0.78", "475"]
    ]
    t7 = Table(t7_data, colWidths=[130, 85, 85, 95, 86.9])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'SegoeUI-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'SegoeUI'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t7)
    story.append(Spacer(1, 4))

    # Figure 2: Confusion Matrix
    fig2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "figure2_confusion_matrix.png")
    if os.path.exists(fig2_path):
        img2 = Image(fig2_path, width=220, height=193)
        img2.hAlign = 'CENTER'
        story.append(img2)
        story.append(Paragraph("Fig. 2. Confusion matrix of CA-Net on the 475 test images (rows: actual, columns: predicted).", caption_style))

    story.append(Paragraph(
        "<b>C. Explainability.</b> Grad-CAM and Grad-CAM++ visualizations are produced for all three classes. Peak intensity falls "
        "on the neuroretinal rim margin and the excavated optic cup rather than on vessels or the image border, which is "
        "consistent with the clinical signs of glaucoma. Grad-CAM++ generally gives tighter localization around the optic disc "
        "region and around subtle notch defects in early-stage cases, which supports its use as the default view in the "
        "interface. These observations come from visual inspection of the heatmaps; no quantitative localization metric was "
        "computed.",
        body_style
    ))
    story.append(Paragraph(
        "<b>D. Caveats.</b> Each model was trained once, so the differences in Table VI carry no confidence interval; on 475 test "
        "images a 2.31-point accuracy gap corresponds to roughly 11 images. Per-class ROC curves and the learned values "
        "of <i>w</i><sub>1</sub>, <i>w</i><sub>2</sub> and <i>w</i><sub>3</sub> are available from the evaluation outputs and the web interface but are not tabulated here.",
        body_style
    ))

    story.append(Paragraph("VIII. Limitations and Future Work", sec_style))
    story.append(Paragraph("The present prototype has several engineering and clinical considerations identified during development:", body_style))
    story.append(Paragraph(
        "1. <b>Portability:</b> Sample and web assets have been configured with relative project paths, while dataset directories "
        "and split generators can be further parameterized via environment variables for cross-platform deployments.",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Reproducibility:</b> All Python dependencies and pinned versions (PyTorch, Torchvision, NumPy &lt;2.0.0, Gradio, "
        "FastAPI, scikit-learn) are documented in <code>requirements.txt</code>; Python 3.8–3.11 is recommended for pre-built wheel compatibility.",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Concurrency:</b> Grad-CAM gradients and activations are stored on the shared model instance, so simultaneous "
        "/api/explain requests can corrupt each other’s heatmaps. Per-request hooks or locking should be used in heavy multi-tenant production.",
        body_style
    ))
    story.append(Paragraph(
        "4. <b>Efficiency:</b> In the local REST dashboard, the client calls /api/predict and /api/explain separately. Returning the "
        "diagnostic prediction alongside heatmaps within a single unified endpoint avoids redundant forward passes.",
        body_style
    ))
    story.append(Paragraph(
        "5. <b>Security:</b> CORS is configured for permissive development access; clinical deployments require strict origin "
        "whitelisting, authenticated session tokens, and input payload sanitization.",
        body_style
    ))

    # ==================== PAGE 7 ====================
    story.append(PageBreak())
    story.append(Paragraph(
        "6. <b>Data:</b> The dataset comprises 1,582 images with 40 from RIM-ONE r1. Distinguishing early glaucoma from physiologically "
        "large cups is challenging without binocular stereoscopic depth or optical coherence tomography (OCT). Class-balanced loss "
        "formulations, heavier spatial augmentations, larger vision backbones, and multi-center testing represent vital next steps.",
        body_style
    ))
    story.append(Paragraph(
        "7. <b>Statistics:</b> Metrics reflect a single complete 50-epoch benchmark run without repeated seed averaging or calibration analysis. "
        "Future work includes stratified 5-fold cross-validation, clinical validation with board-certified ophthalmologists, and "
        "ONNX/INT8 quantization for edge deployment on low-cost fundus cameras.",
        body_style
    ))

    story.append(Paragraph("IX. Conclusion", sec_style))
    story.append(Paragraph(
        "This paper presented GlaucoVision, a complete pipeline for three-stage glaucoma classification built on CA-Net with "
        "a DenseNet-121 backbone. Under an identical 50-epoch protocol, the attention module lifts test accuracy from "
        "74.74% to 77.05%, macro F1 from 69.33% to 74.17%, macro recall from 70.30% to 76.05% and AUC from 0.897 to "
        "0.911 on a held-out set of 475 images. Advanced-stage glaucoma is detected reliably (recall 0.88), while Early-stage "
        "remains the main weakness (F1 0.58). The decisions are made inspectable through Grad-CAM and Grad-CAM++ "
        "overlays within both a lightweight FastAPI web application and a public Hugging Face Spaces cloud instance. "
        "Addressing the portability, concurrency and security aspects identified above, and validating on multi-center external cohorts, "
        "will establish a resilient foundation for real-world automated glaucoma screening support.",
        body_style
    ))

    story.append(Paragraph("Appendix A. Repository Layout and Reproduction", sec_style))
    story.append(Paragraph(
        "The CA-Net model was trained with the command below; the baseline uses the same protocol with the attention "
        "module disabled (checkpoint suffix _none). Evaluation produces the confusion matrix, ROC curves and Grad-CAM "
        "figures, and run_app.py starts the local dashboard on port 8000.",
        body_style
    ))
    
    code_text = (
        "<b>Training Command:</b><br/>"
        "python train.py --dataset lmg --backbone densenet121 --attention cam --epochs 50 --batch_size 16 --lr 0.005<br/><br/>"
        "<b>Repository Structure & Assets:</b><br/>"
        "GlaucoVision/<br/>"
        "&nbsp;&nbsp;dataset/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prepare_lmg.py (Harvard + RIM-ONE, 70:30 splits), dataset.py (loader, augment)<br/>"
        "&nbsp;&nbsp;models/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tcab.py, sab.py, cam.py (TCAB -&gt; SAB), ca_net.py (backbone + CAM + head)<br/>"
        "&nbsp;&nbsp;explainability/&nbsp;&nbsp;&nbsp;gradcam.py (Grad-CAM and Grad-CAM++ implementations)<br/>"
        "&nbsp;&nbsp;data_splits/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lmg_train.csv (774), lmg_val.csv (333), lmg_test.csv (475), lmg_all.csv (1,582)<br/>"
        "&nbsp;&nbsp;checkpoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_lmg_densenet121_cam.pth (~86 MB), best_lmg_densenet121_none.pth (~84 MB)<br/>"
        "&nbsp;&nbsp;evaluation_results/ confusion_matrix.png, roc_curves.png, gradcam_visualizations.png<br/>"
        "&nbsp;&nbsp;app/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;backend.py (FastAPI), static/ (index.html, style.css, app.js)<br/>"
        "&nbsp;&nbsp;app.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Hugging Face Spaces ZeroGPU native Gradio web application<br/>"
        "&nbsp;&nbsp;requirements.txt&nbsp;&nbsp;Pinned production dependencies for complete reproduction<br/>"
        "&nbsp;&nbsp;train.py evaluate.py run_app.py test_api.py<br/><br/>"
        "<b>Live Deployments:</b><br/>"
        "• Hugging Face Spaces: <u>https://huggingface.co/spaces/Aayu62/GlaucoVision</u><br/>"
        "• GitHub Repository: <u>https://github.com/Aayu62/GlaucoVision</u>"
    )
    story.append(Paragraph(code_text, code_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("References", sec_style))
    refs_p7 = [
        "[1] Y.-C. Tham, X. Li, T. Y. Wong, H. A. Quigley, T. Aung, and C.-Y. Cheng, “Global prevalence of glaucoma and projections of glaucoma burden through 2040: A systematic review and meta-analysis,” <i>Ophthalmology</i>, vol. 121, no. 11, pp. 2081–2090, 2014.",
        "[2] H. Fu, J. Cheng, Y. Xu, C. Zhang, D. W. K. Wong, J. Liu, and X. Cao, “Disc-aware ensemble network for glaucoma screening from fundus image,” <i>IEEE Trans. Med. Imag.</i>, vol. 37, no. 11, pp. 2493–2501, Nov. 2018.",
        "[3] J. M. Ahn, S. Kim, K.-S. Ahn, S.-H. Cho, K. B. Lee, and U. S. Kim, “A deep learning model for the detection of both advanced and early glaucoma using fundus photography,” <i>PLoS ONE</i>, vol. 13, no. 11, Art. no. e0207982, Nov. 2018.",
        "[4] D. Das, D. R. Nayak, and R. B. Pachori, “CA-Net: A novel cascaded attention-based network for multistage glaucoma classification using fundus images,” <i>IEEE Trans. Instrum. Meas.</i>, vol. 72, Art. no. 2531110, pp. 1–10, 2023.",
        "[5] S. Dua, U. R. Acharya, P. Chowriappa, and S. V. Sree, “Wavelet-based energy features for glaucomatous image classification,” <i>IEEE Trans. Inf. Technol. Biomed.</i>, vol. 16, no. 1, pp. 80–87, Jan. 2012.",
        "[6] D. Parashar and D. Agrawal, “2-D compact variational mode decomposition-based automatic classification of glaucoma stages from fundus images,” <i>IEEE Trans. Instrum. Meas.</i>, vol. 70, pp. 1–10, 2021."
    ]
    for r in refs_p7:
        story.append(Paragraph(r, ref_style))

    # ==================== PAGE 8 ====================
    story.append(PageBreak())
    refs_p8 = [
        "[7] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, Jun. 2018, pp. 7132–7141.",
        "[8] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” in <i>Proc. Eur. Conf. Comput. Vis. (ECCV)</i>, 2018, pp. 3–19.",
        "[9] D. Misra, T. Nalamada, A. U. Arasanipalai, and Q. Hou, “Rotate to attend: Convolutional triplet attention module,” in <i>Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)</i>, Jan. 2021, pp. 3138–3147.",
        "[10] A. He, T. Li, N. Li, K. Wang, and H. Fu, “CABNet: Category attention block for imbalanced diabetic retinopathy grading,” <i>IEEE Trans. Med. Imag.</i>, vol. 40, no. 1, pp. 143–153, Jan. 2021.",
        "[11] H. Tian, S. Lu, Y. Sun, and H. Li, “GC-Net: Global and class attention blocks for automated glaucoma classification,” in <i>Proc. IEEE 17th Conf. Ind. Electron. Appl. (ICIEA)</i>, Dec. 2022, pp. 498–503.",
        "[12] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in <i>Proc. IEEE Int. Conf. Comput. Vis. (ICCV)</i>, Oct. 2017, pp. 618–626.",
        "[13] A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, “Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks,” in <i>Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)</i>, Mar. 2018, pp. 839–847.",
        "[14] F. Fumero, S. Alayon, J. L. Sanchez, J. Sigut, and M. Gonzalez-Hernandez, “RIM-ONE: An open retinal image database for optic nerve evaluation,” in <i>Proc. 24th Int. Symp. Comput.-Based Med. Syst. (CBMS)</i>, Jun. 2011, pp. 1–6.",
        "[15] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, Jul. 2017, pp. 2261–2269.",
        "[16] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in <i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, 2019, pp. 8024–8035."
    ]
    for r in refs_p8:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Built PDF successfully: {filename}")

if __name__ == "__main__":
    build_pdf("GlaucoVision_Report.pdf")
    print("GlaucoVision_Report.pdf generated and updated successfully!")
