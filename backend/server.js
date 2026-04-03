require('dotenv').config()

const express = require('express')
const sql = require('mssql')
const cors = require('cors')
const { spawn } = require('child_process')
const path = require('path')
const OpenAI = require("openai")

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})


const app = express()
app.use(cors({
  origin: "*",
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"]
}))

const config = {
  user: 'almighty',
  password: 'almighty',
  server: '127.0.0.1',
  port: 22020,
  database: 'SmartdoctorDB',
  options: {
    encrypt: false,
    trustServerCertificate: true
  }
}

// AI 실행 API (최종 완성)
app.get('/run-ai', async (req, res) => {
  try {
    const pool = await sql.connect(config)

    // 환자 조회
    const result = await pool.request().query(`
      SELECT 
        CUST_NO,
        방문간격 as gap_days,
        방문횟수 as visit_count,
        평균방문간격 as avg_gap_days,
        최근방문여부 as recent_visit,
        장기미방문여부 as long_term_no_visit,
        이전방문간격 as prev_gap,
        방문간격변화량 as gap_change,
        방문지연여부 as delay_flag,
        최근3회평균간격 as recent3_avg,
        방문간격비율 as gap_ratio
      FROM patient_summary
    `)

    const patients = result.recordset
    console.log(`총 환자 수: ${patients.length}`)

    // Python batch 실행
    const pythonFile = path.join(__dirname, '..', 'ai', 'predict_batch.py')

    const py = spawn('python', [pythonFile])

    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    py.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    py.stdin.write(JSON.stringify(patients))
    py.stdin.end()

    py.on('close', async (code) => {
    

      if (code !== 0) {
        console.error(' Python 에러:', stderr)
        return res.status(500).send(stderr)
      }

      let predictions
      try {
        predictions = JSON.parse(stdout)
      } catch (e) {
        console.error(' JSON 파싱 실패:', stdout)
        return res.status(500).send('JSON parse error')
      }

      console.log(`예측 완료: ${predictions.length}`)

      // 일반 테이블 생성 (temp X)
      await pool.request().query(`
        IF OBJECT_ID('temp_predictions', 'U') IS NOT NULL
          DROP TABLE temp_predictions

        CREATE TABLE temp_predictions (
          CUST_NO NVARCHAR(50),
          revisit_prob FLOAT
        )
      `)

      // bulk insert
      const table = new sql.Table('temp_predictions')
      table.create = false
      
      table.columns.add('CUST_NO', sql.NVarChar(50))
      table.columns.add('revisit_prob', sql.Float)

      predictions.forEach(p => {
        table.rows.add(p.CUST_NO, parseFloat(p.revisit_prob))
      })

      await pool.request().bulk(table)

      // 한번에 UPDATE
      await pool.request().query(`
        UPDATE p
        SET p.revisit_prob = t.revisit_prob
        FROM patient_summary p
        JOIN temp_predictions t
        ON LTRIM(RTRIM(p.CUST_NO)) = LTRIM(RTRIM(t.CUST_NO))
      `)

      // 테이블 삭제
      // await pool.request().query(`DROP TABLE temp_predictions`)

      console.log(' DB 업데이트 완료')

      res.send(' AI 배치 업데이트 완료')

    })

  } catch (err) {
    console.error(' 서버 에러:', err)
    res.status(500).send(err.message || 'Error')
  }
})

// 프론트용 API
app.get('/patient-risk', async (req, res) => {
  try {
    const pool = await sql.connect(config)

    const result = await pool.request().query(`
     SELECT 
    CUST_NO,
    NAME as name,
    방문횟수 as visit_count,
    평균방문간격 as avg_gap_days,
    최근방문여부 as recent_visit,     
    방문지연여부 as delay_flag, 
    last_visit_day,
    days_since_last_visit,
    revisit_prob,

    -- 총점 계산
    (
        -- 1. 모델 점수
        (CASE 
            WHEN revisit_prob <= 0.3 THEN 3
            WHEN revisit_prob <= 0.6 THEN 2
            WHEN revisit_prob <= 0.8 THEN 1
            ELSE 0
        END)

        +

        -- 2. 경과일 점수
        (CASE 
            WHEN days_since_last_visit >= 평균방문간격 * 3 THEN 3
            WHEN days_since_last_visit >= 평균방문간격 * 2 THEN 2
            WHEN days_since_last_visit >= 평균방문간격 * 1.5 THEN 1
            ELSE 0
        END)

        +

        -- 3. 강제 이탈 (하드 룰)
        (CASE 
            WHEN days_since_last_visit >= 365 THEN 3
            WHEN days_since_last_visit >= 180 THEN 2
            ELSE 0
        END)

        +

        -- 4. 방문 지연
        (CASE 
            WHEN 방문지연여부 = 1 THEN 1
            ELSE 0
        END)
    ) as total_score,

    -- 최종 위험도
    CASE 
        WHEN 
            (
                (CASE 
                    WHEN revisit_prob <= 0.3 THEN 3
                    WHEN revisit_prob <= 0.6 THEN 2
                    WHEN revisit_prob <= 0.8 THEN 1
                    ELSE 0
                END)
                +
                (CASE 
                    WHEN days_since_last_visit >= 평균방문간격 * 3 THEN 3
                    WHEN days_since_last_visit >= 평균방문간격 * 2 THEN 2
                    WHEN days_since_last_visit >= 평균방문간격 * 1.5 THEN 1
                    ELSE 0
                END)
                +
                (CASE 
                    WHEN days_since_last_visit >= 365 THEN 3
                    WHEN days_since_last_visit >= 180 THEN 2
                    ELSE 0
                END)
                +
                (CASE 
                    WHEN 방문지연여부 = 1 THEN 1
                    ELSE 0
                END)
            ) >= 6 THEN 'HIGH'

        WHEN 
            (
                (CASE 
                    WHEN revisit_prob <= 0.3 THEN 3
                    WHEN revisit_prob <= 0.6 THEN 2
                    WHEN revisit_prob <= 0.8 THEN 1
                    ELSE 0
                END)
                +
                (CASE 
                    WHEN days_since_last_visit >= 평균방문간격 * 3 THEN 3
                    WHEN days_since_last_visit >= 평균방문간격 * 2 THEN 2
                    WHEN days_since_last_visit >= 평균방문간격 * 1.5 THEN 1
                    ELSE 0
                END)
                +
                (CASE 
                    WHEN days_since_last_visit >= 365 THEN 3
                    WHEN days_since_last_visit >= 180 THEN 2
                    ELSE 0
                END)
                +
                (CASE 
                    WHEN 방문지연여부 = 1 THEN 1
                    ELSE 0
                END)
            ) >= 3 THEN 'MID'

        ELSE 'LOW'
    END as risk_level

FROM patient_summary
    `)

    res.json(result.recordset)

  } catch (err) {
    console.error(err)
    res.status(500).send('Error')
  }
})

app.get('/patients', async (req, res) => {
  try {
    const pool = await sql.connect(config)
    const name = req.query.name || ''
    const page = parseInt(req.query.page) || 1
    const pageSize = 50
    const offset = (page - 1) * pageSize

    const result = await pool.request()
      .input('name', `%${name}%`)
      .query(`
        SELECT 
          CUST_NO,
          NAME
        FROM CUST_INFO_리버스홍대
        WHERE NAME LIKE @name
        OR CUST_NO LIKE @name
        ORDER BY NAME
        OFFSET ${offset} ROWS FETCH NEXT ${pageSize} ROWS ONLY
      `)

    res.json(result.recordset)

  } catch (err) {
    console.error(err)
    res.status(500).send('error')
  }
})

app.get('/highvalue-detail/:cust_no', async (req, res) => {
  try {
    const pool = await sql.connect(config)
    const custNo = req.params.cust_no

    // 🔹 summary
    const summary = await pool.request()
      .input('cust_no', custNo)
      .query(`
        SELECT 
          cust_no,
          name,
          total_amt,
          avg_amt,
          tier,
          last_mdcl_day
        FROM patient_amount_summary
        WHERE cust_no = @cust_no
      `)

    // 🔹 daily
    const daily = await pool.request()
      .input('cust_no', custNo)
      .query(`
        SELECT mdcl_day, daily_amt
        FROM patient_amount_daily
        WHERE cust_no = @cust_no
        ORDER BY mdcl_day
      `)

    res.json({
      summary: summary.recordset[0],
      daily: daily.recordset
    })

  } catch (err) {
    console.error(err)
    res.status(500).send('error')
  }
})

app.use(express.json())
app.post('/explain', async (req, res) => {
  try {
    const row = req.body

    const prompt = `
너는 병원 환자의 재방문 가능성을 분석하는 데이터 분석 전문가이다.

다음 기준을 기반으로 환자의 상태를 해석하라.

[판단 기준]
- 경과일 > 평균방문간격 → 방문이 지연된 상태 (위험 신호)
- 경과일 > 평균방문간격의 1.5배 → 명확한 지연 상태 (이탈 위험 증가)
- 경과일 > 평균방문간격의 2배 → 매우 높은 이탈 위험
- 최근방문여부 = 0 → 최근 방문 없음 (위험 증가)
- 방문지연여부 = 1 → 패턴 붕괴 (중요 위험 신호)

이 기준을 반드시 활용하여 설명하라.

[환자 데이터]
- 방문간격: ${row.gap_days}일
- 방문횟수: ${row.visit_count}회
- 평균방문간격: ${row.avg_gap_days}일
- 최근방문여부: ${row.recent_visit}
- 장기미방문여부: ${row.long_term_no_visit}
- 이전방문간격: ${row.prev_gap}
- 방문간격변화량: ${row.gap_change}
- 방문지연여부: ${row.delay_flag}
- 최근3회평균간격: ${row.recent3_avg}
- 방문간격비율: ${row.gap_ratio}

[현재 상태]
- 경과일: ${row.days_since_last_visit}일
- 예측 확률: ${row.revisit_prob}

[위험 점수]
- 총점: ${row.total_score}
- 위험도: ${row.risk_level}

요구사항:
1. 반드시 위 판단 기준을 근거로 설명할 것
2. 핵심 원인 2~3개만 설명
3. 번호(1,2,3) 형식으로 작성
4. 각 문장은 줄바꿈으로 구분
5. "이 환자는 ~ 때문에 ~로 판단됩니다" 형식으로 작성
6. 점수(total_score)가 높아진 원인을 함께 설명할 것
7. 데이터 기반으로 논리적으로 설명
`

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "너는 데이터 분석 전문가이다." },
        { role: "user", content: prompt }
      ],
      temperature: 0.3
    })

    const explanation = response.choices[0].message.content

    res.json({ explanation })

  } catch (err) {
    console.error(err)
    res.status(500).send('GPT 오류')
  }
})

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/main.html'));
});

app.use(express.static(path.join(__dirname, '../frontend')));


app.listen(3000, () => {
  console.log('🔥 Server running on 3000')
})
