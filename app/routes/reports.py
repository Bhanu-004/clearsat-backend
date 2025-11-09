# backend/app/routes/reports.py
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import Response, StreamingResponse
from bson import ObjectId
import io
import asyncio
import aiofiles
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import base64
from datetime import datetime
import tempfile
import os
import logging
from typing import Dict, Any

from app.auth.auth import get_current_active_user, require_role
from app.models.user import UserRole
from app.database import get_analysis_collection, get_report_collection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])

async def generate_pdf_report_async(analysis_data: dict) -> bytes:
    """Generate PDF report asynchronously to avoid blocking"""
    return await asyncio.get_event_loop().run_in_executor(
        None, 
        generate_pdf_report_sync,
        analysis_data
    )

def generate_pdf_report_sync(analysis_data: dict) -> bytes:
    """Synchronous PDF generation with enhanced features and error handling"""
    buffer = io.BytesIO()
    
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Enhanced custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            textColor=colors.HexColor('#1E6EA7'),
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#2E86AB'),
            borderPadding=5,
            backColor=colors.HexColor('#F8F9FA')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#4A6572')
        )
        
        story = []
        
        # Header with logo and title
        header_table = Table([
            [Paragraph("ClearSat", ParagraphStyle('Header', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1E6EA7'))),
             Paragraph("Satellite Analysis Report", ParagraphStyle('Header', parent=styles['Normal'], fontSize=12, alignment=2))]
        ], colWidths=[3*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 20))
        
        # Title
        story.append(Paragraph("Environmental Analysis Report", title_style))
        story.append(Spacer(1, 10))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = (
            f"This comprehensive report presents satellite imagery analysis for <b>{analysis_data['location']['name']}</b> "
            f"conducted between <b>{analysis_data['start_date']}</b> and <b>{analysis_data['end_date']}</b>. "
            "The analysis provides detailed insights into environmental conditions, vegetation health, "
            "and land cover changes using advanced remote sensing techniques."
        )
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Location Details
        story.append(Paragraph("Location Details", heading_style))
        location_info = [
            ["Location Name", analysis_data['location']['name']],
            ["Coordinates", f"{analysis_data['location']['latitude']:.4f}°N, {analysis_data['location']['longitude']:.4f}°E"],
            ["State/District", f"{analysis_data['location'].get('state', 'N/A')} / {analysis_data['location'].get('district', 'N/A')}"],
            ["Analysis Period", f"{analysis_data['start_date']} to {analysis_data['end_date']}"],
            ["Satellite Source", analysis_data['satellite_source'].upper()],
            ["Analysis Type", analysis_data['analysis_type']],
            ["Analysis Radius", f"{analysis_data.get('buffer_km', 10)} km"]
        ]
        
        location_table = Table(location_info, colWidths=[2.2*inch, 3.3*inch])
        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F7FA')])
        ]))
        story.append(location_table)
        story.append(Spacer(1, 20))
        
        # Key Insights
        if analysis_data.get('results') and analysis_data['results'].get('insights'):
            story.append(Paragraph("Key Insights", heading_style))
            for insight in analysis_data['results']['insights']:
                # Add appropriate emoji based on insight type
                emoji = "📊"
                if "excellent" in insight.lower() or "ideal" in insight.lower():
                    emoji = "✅"
                elif "warning" in insight.lower() or "consider" in insight.lower():
                    emoji = "⚠️"
                elif "water" in insight.lower():
                    emoji = "💧"
                elif "vegetation" in insight.lower():
                    emoji = "🌿"
                
                story.append(Paragraph(f"{emoji} {insight}", styles['Normal']))
                story.append(Spacer(1, 5))
            story.append(Spacer(1, 15))
        
        # Statistics Section
        if analysis_data.get('results') and analysis_data['results'].get('statistics'):
            story.append(Paragraph("Statistical Summary", heading_style))
            stats = analysis_data['results']['statistics']
            
            # Main statistics table
            stats_data = [
                ["Metric", "Value", "Interpretation"],
                ["Mean", f"{stats.get('mean', 0):.4f}", get_stat_interpretation(analysis_data['analysis_type'], stats.get('mean', 0))],
                ["Median", f"{stats.get('median', 0):.4f}", "Middle value of dataset"],
                ["Std Deviation", f"{stats.get('std', 0):.4f}", "Data variability measure"],
                ["Minimum", f"{stats.get('min', 0):.4f}", "Lowest observed value"],
                ["Maximum", f"{stats.get('max', 0):.4f}", "Highest observed value"],
                ["Data Points", str(stats.get('count', 0)), "Total observations"]
            ]
            
            # Add additional stats if available
            if 'range' in stats:
                stats_data.append(["Range", f"{stats.get('range', 0):.4f}", "Value spread"])
            if 'q1' in stats and 'q3' in stats:
                stats_data.append(["IQR", f"{stats.get('q3', 0) - stats.get('q1', 0):.4f}", "Middle 50% spread"])
            
            stats_table = Table(stats_data, colWidths=[1.5*inch, 1.2*inch, 2.8*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F7FA')]),
                ('WORDWRAP', (0, 0), (-1, -1), True)
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 20))
        
        # Time Series Data (if available)
        if (analysis_data.get('results') and 
            analysis_data['results'].get('time_series') and 
            len(analysis_data['results']['time_series']) > 1):
            
            story.append(Paragraph("Temporal Analysis", heading_style))
            time_series = analysis_data['results']['time_series']
            
            # Create a simple time series table (first 5 entries)
            preview_data = [["Date", "Value"]] + [
                [ts['date'], f"{ts['value']:.4f}"] 
                for ts in time_series[:5]
            ]
            
            if len(time_series) > 5:
                preview_data.append(["...", f"... ({len(time_series) - 5} more entries)"])
            
            ts_table = Table(preview_data, colWidths=[2*inch, 1.5*inch])
            ts_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A6572')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(ts_table)
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"Total time series entries: {len(time_series)}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Methodology
        story.append(Paragraph("Methodology", heading_style))
        methodology_text = (
            "This analysis utilizes satellite imagery from approved sources with advanced remote sensing algorithms. "
            "Data processing includes cloud masking, atmospheric correction, and temporal compositing to ensure accuracy. "
            "All calculations follow established scientific methodologies for environmental monitoring."
        )
        story.append(Paragraph(methodology_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Footer with enhanced information
        footer_text = (
            f"Report generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')} | "
            f"ClearSat Satellite Analysis Platform v1.0 | "
            f"Confidential - For authorized use only"
        )
        
        story.append(Paragraph(
            footer_text,
            ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=7,
                textColor=colors.grey,
                alignment=1
            )
        ))
        
        # Build document
        doc.build(story)
        buffer.seek(0)
        
        # Validate PDF size
        pdf_content = buffer.getvalue()
        if len(pdf_content) < 1000:  # Too small, likely error
            raise ValueError("Generated PDF is too small, likely generation error")
            
        return pdf_content
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        # Return a simple error PDF
        return generate_error_pdf(str(e))

def generate_error_pdf(error_message: str) -> bytes:
    """Generate a simple error PDF when main generation fails"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "ClearSat - Report Generation Error")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "We encountered an error while generating your report:")
    
    c.setFont("Helvetica", 10)
    # Wrap error message
    y_position = 670
    for line in wrap_text(error_message, 80):
        c.drawString(100, y_position, line)
        y_position -= 15
    
    c.drawString(100, 600, "Please try again or contact support if the issue persists.")
    c.drawString(100, 580, f"Error time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def wrap_text(text: str, width: int) -> list:
    """Wrap text to specified width"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_stat_interpretation(analysis_type: str, value: float) -> str:
    """Get interpretation for statistical values"""
    if analysis_type == "NDVI":
        if value > 0.6: return "Excellent vegetation"
        elif value > 0.3: return "Good vegetation"
        elif value > 0.1: return "Sparse vegetation"
        else: return "Low vegetation"
    elif analysis_type == "EVI":
        if value > 0.5: return "Excellent vigor"
        elif value > 0.2: return "Good condition"
        else: return "Low to moderate"
    elif analysis_type == "NDWI":
        if value > 0.2: return "High water content"
        elif value > 0: return "Moderate moisture"
        else: return "Dry conditions"
    else:
        return "Standard metric"

@router.post("/{analysis_id}/pdf")
async def generate_analysis_pdf(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user)
):
    """Generate PDF report for analysis with enhanced features"""
    
    # Check if user is guest
    if current_user.get("role") == UserRole.GUEST:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Guest users cannot export PDF reports",
                "action": "Please register for full access to export capabilities"
            }
        )
    
    analysis_collection = get_analysis_collection()
    report_collection = get_report_collection()
    
    try:
        # Validate analysis ID
        if not ObjectId.is_valid(analysis_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid analysis ID format"
            )
        
        # Find analysis
        analysis = await analysis_collection.find_one({
            "_id": ObjectId(analysis_id),
            "user_id": str(current_user["_id"])
        })
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        if analysis.get("status") != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis must be completed to generate report"
            )
        
        # Check if report already exists
        existing_report = await report_collection.find_one({
            "analysis_id": analysis_id,
            "user_id": str(current_user["_id"])
        })
        
        if existing_report and existing_report.get('pdf_content'):
            # Return existing report
            pdf_content = existing_report['pdf_content']
            logger.info(f"📄 Returning cached PDF for analysis {analysis_id}")
        else:
            # Generate new PDF asynchronously
            logger.info(f"🔄 Generating new PDF for analysis {analysis_id}")
            pdf_content = await generate_pdf_report_async(analysis)
            
            # Store report in database for caching
            report_data = {
                "analysis_id": analysis_id,
                "user_id": str(current_user["_id"]),
                "pdf_content": pdf_content,
                "generated_at": datetime.utcnow(),
                "file_size": len(pdf_content)
            }
            
            await report_collection.update_one(
                {"analysis_id": analysis_id, "user_id": str(current_user["_id"])},
                {"$set": report_data},
                upsert=True
            )
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        location_name = analysis['location']['name'].replace(' ', '_')
        filename = f"clearsat_report_{location_name}_{timestamp}.pdf"
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_content)),
                "X-Report-Generated": datetime.utcnow().isoformat(),
                "X-Analysis-Type": analysis.get('analysis_type', 'unknown')
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation failed for analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF report"
        )

@router.get("/{analysis_id}/status")
async def get_report_status(
    analysis_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Check if PDF report exists for analysis"""
    report_collection = get_report_collection()
    
    try:
        if not ObjectId.is_valid(analysis_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid analysis ID format"
            )
        
        report = await report_collection.find_one({
            "analysis_id": analysis_id,
            "user_id": str(current_user["_id"])
        })
        
        if report:
            return {
                "report_exists": True,
                "generated_at": report.get("generated_at"),
                "file_size": report.get("file_size"),
                "ready_for_download": True
            }
        else:
            return {
                "report_exists": False,
                "ready_for_download": False
            }
            
    except Exception as e:
        logger.error(f"Failed to check report status for {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check report status"
        )