# Emergency Alert System Setup Guide

## Overview
The Emergency Alert System automatically sends SMS notifications to emergency personnel (Fire Department, Police, Medical Services, Security, Management) when dangerous crowd conditions are detected by the YOLO-based crowd monitoring system.

## Prerequisites

### 1. Twilio Account Setup
1. **Create Account**: Go to [twilio.com](https://twilio.com) and sign up for a free account
2. **Verify Phone Number**: Complete phone verification during signup
3. **Get Phone Number**: Purchase a Twilio phone number from the console
4. **Find Credentials**: In the Twilio Console dashboard, locate:
   - Account SID (starts with "AC...")
   - Auth Token (click the eye icon to reveal)
   - Your Twilio Phone Number (in E.164 format: +1234567890)

### 2. Environment Variables Setup

#### Windows (Command Prompt)
```cmd
set TWILIO_ACCOUNT_SID=your_account_sid_here
set TWILIO_AUTH_TOKEN=your_auth_token_here
set TWILIO_NUMBER=+1234567890
```

#### Windows (PowerShell)
```powershell
$env:TWILIO_ACCOUNT_SID="your_account_sid_here"
$env:TWILIO_AUTH_TOKEN="your_auth_token_here"
$env:TWILIO_NUMBER="+1234567890"
```

#### Linux/Mac
```bash
export TWILIO_ACCOUNT_SID=your_account_sid_here
export TWILIO_AUTH_TOKEN=your_auth_token_here
export TWILIO_NUMBER=+1234567890
```

#### Permanent Setup (.env file)
Create a `.env` file in the project root:
```env
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_NUMBER=+1234567890
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Emergency Contacts Setup
1. Navigate to **Emergency Alerts** page in the dashboard
2. Add emergency contacts for each department:

#### Recommended Contacts:
- **Fire Department**: Local fire emergency line
- **Police Department**: Local police emergency line  
- **Medical Emergency**: Local EMS/ambulance service
- **Security Team**: Internal security personnel
- **Facility Management**: Building/event management

#### Contact Information Required:
- Name (e.g., "Fire Department Emergency")
- Phone Number (E.164 format: +1234567890)
- Department Type
- Priority Level (1 = highest priority)
- Active Status

### 2. Alert Configuration
Configure alert behavior in the Emergency Alerts page:

- **Enable Alert System**: Turn on/off SMS notifications
- **Retry Attempts**: Number of retry attempts for failed messages (default: 3)
- **Retry Delay**: Minutes between retry attempts (default: 2)
- **Auto Escalate**: Automatically escalate unacknowledged critical alerts
- **Escalation Delay**: Minutes before escalation (default: 5)
- **Include Location**: Add location info in alert messages
- **Include Timestamp**: Add timestamp in alert messages

### 3. Crowd Monitor Integration
In the Enhanced Crowd Monitor page:

1. **Enable SMS Alerts**: Check the checkbox in sidebar
2. **Alert Cooldown**: Set minimum minutes between alerts for same zone (default: 5)
3. **Auto Escalate Critical**: Enable automatic escalation for critical alerts

## Alert Levels & Routing

### Alert Thresholds (configurable)
- **Safe**: ≤ 4.2 people/sqm (no alerts)
- **Warning**: 4.2-5.4 people/sqm → Security + Management
- **Critical**: > 6.0 people/sqm → Fire + Police + Medical + Security + Management

### Department Routing
- **LOW**: Security only
- **MEDIUM**: Security + Management
- **HIGH**: Security + Management + Fire
- **CRITICAL**: All departments (Fire + Police + Medical + Security + Management)

## Alert Message Format

### Example Critical Alert:
```
🚨 CRITICAL EMERGENCY ALERT

⏰ Time: 2025-08-23 10:06:08
📍 Location: Monitoring Zone: Main Entrance
📋 Details: CROWD DENSITY ALERT: ⚠ Overcrowding at Main Entrance! Density: 6.50 people/sqm. Immediate attention required.

📊 Additional Information:
• People Count: 65
• Total Objects: 73
• Vehicles Present: 5
• Zone Area: 1000 sqm
• Frame Number: 150
• Detection Confidence: High

⚠️ This is an automated emergency alert from the Crowd Monitoring System.
🔄 Reply 'RECEIVED' to acknowledge this alert.
```

## Testing the System

### 1. Test Twilio Connection
1. Go to **Emergency Alerts** → **Twilio Setup** tab
2. Click "🔗 Test Twilio Connection"
3. Verify successful connection message

### 2. Send Test Alert
1. Go to **Emergency Alerts** → **Test System** tab
2. Configure test parameters:
   - Alert Level (start with LOW)
   - Test Message
   - Target Departments
   - Location
3. Click "🧪 Send Test Alert"
4. Verify SMS delivery to configured contacts

### 3. Test Crowd Monitor Integration
1. Go to **Enhanced Crowd Monitor**
2. Enable SMS Alerts in sidebar
3. Upload a video with crowd scenes
4. Adjust density thresholds to trigger alerts
5. Verify alerts are sent when thresholds exceeded

## Troubleshooting

### Common Issues

#### 1. "Twilio client not initialized"
- **Cause**: Missing or incorrect environment variables
- **Solution**: Verify TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER are set correctly

#### 2. "Failed to send SMS"
- **Cause**: Invalid phone numbers, insufficient Twilio credits, or network issues
- **Solution**: 
  - Verify phone numbers in E.164 format (+1234567890)
  - Check Twilio account balance
  - Verify Twilio phone number is active

#### 3. "No active contacts found"
- **Cause**: No emergency contacts configured or all contacts disabled
- **Solution**: Add emergency contacts in Emergency Alerts page

#### 4. "Alert system disabled"
- **Cause**: Alert system turned off in configuration
- **Solution**: Enable alert system in Emergency Alerts → Alert Settings

### Error Logs
Check Streamlit console output for detailed error messages:
```bash
streamlit run streamlit_app.py
```

## Security Considerations

### 1. Credential Protection
- Never commit Twilio credentials to version control
- Use environment variables or secure credential management
- Rotate auth tokens periodically

### 2. Phone Number Verification
- Verify all emergency contact phone numbers
- Test message delivery regularly
- Keep contact information updated

### 3. Alert Rate Limiting
- Configure appropriate cooldown periods
- Monitor alert frequency to prevent spam
- Set up escalation procedures for system failures

## Maintenance

### Regular Tasks
1. **Monthly**: Test all emergency contacts
2. **Quarterly**: Review and update contact information
3. **Annually**: Review alert thresholds and procedures

### Monitoring
- Check alert history regularly
- Monitor success/failure rates
- Verify escalation procedures work correctly

## Support

### Twilio Support
- [Twilio Documentation](https://www.twilio.com/docs)
- [Twilio Console](https://console.twilio.com)
- [Twilio Support](https://support.twilio.com)

### System Logs
Alert activity is logged in:
- Streamlit console output
- Alert history in Emergency Alerts page
- CSV export files from crowd monitoring

## Emergency Procedures

### If Alert System Fails
1. Check Twilio service status
2. Verify environment variables
3. Test with manual alert from Emergency Alerts page
4. Use backup communication methods
5. Contact system administrator

### False Alert Procedures
1. Acknowledge alert immediately in system
2. Notify emergency contacts of false alarm
3. Review and adjust alert thresholds
4. Document incident for system improvement
